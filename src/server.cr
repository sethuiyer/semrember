# server.cr — the semrember service.
#
#   semrember-server [--model model.amem] [--static static.bin] [--port 7379]
#
# A Redis-shaped HTTP/JSON API over a compressed associative memory.
#
# The split matters and is deliberate:
#
#   exact     GET / PUT / DELETE / EXISTS / SCAN operate on the record
#             store and NEVER invoke the model.  A GET is a hash lookup.
#   semantic  SIMILAR / ESCALATE / NOVELTY invoke the model and return
#             ranked candidates, scores and confidence — never a bare
#             answer, because several records may share one.
#
# Redis compatibility is a mental model, not a wire protocol: the transport
# is HTTP+JSON, the command names are borrowed so the concepts transfer, and
# the semantic layer has no Redis equivalent at all.

require "http/server"
require "json"
require "./semrember"
require "./csv"

module Semrember
  class Server
    getter records : Hash(String, String)
    getter expiry : Hash(String, Time)

    @mem : Memory
    @lock : Mutex
    @started : Time

    def initialize(model : String?, static : String?, train : String? = nil,
                   key_col : String = "message", val_col : String = "action")
      encoder = static.try { |p| StaticEncoder.new(p) }
      @mem = if model
               Memory.load(model, static: encoder)
             elsif train
               Memory.new(solver: :dual, static: encoder).fit(
                 Csv.load_pairs(train.not_nil!, key_col, val_col).to_h)
             else
               Memory.new(solver: :dual, static: encoder)
             end
      @records = {} of String => String
      @expiry = {} of String => Time
      @lock = Mutex.new
      @started = Time.utc
      # A model loaded from disk has no training keys — the file format does
      # not persist them — so the exact record store starts empty and the
      # similarity layer is unavailable.  A model trained here has both.
      if train
        Csv.load_pairs(train.not_nil!, key_col, val_col).each do |(k, v)|
          @records[k] = v
        end
      end
    end

    def listen(host : String, port : Int32)
      puts "semrember #{Semrember::VERSION} listening on http://#{host}:#{port}"
      puts "  exact:    GET PUT DELETE /v1/kv/{key}"
      puts "  semantic: POST /v1/similar  POST /v1/escalate  GET /v1/novelty"
      # `listen` binds and then runs the accept loop; `bind_tcp` alone only
      # binds and returns.
      # The server already gives each connection its own fiber; spawning
      # here would let the handler return (and the response close) before
      # the work ran.
      HTTP::Server.new do |ctx|
        handle(ctx)
      end.listen(host, port)
    end

    # ── routing ─────────────────────────────────────────────────────────
    private def handle(ctx : HTTP::Server::Context)
      ctx.response.content_type = "application/json"
      path = ctx.request.path
      method = ctx.request.method
      begin
        if path == "/v1/ping"
          respond(ctx, 200) { |j| j.object { j.field "ok", true; j.field "version", Semrember::VERSION } }
          return
        end
        if path == "/v1/info"
          respond(ctx, 200) { |j| j.object { info_json(j) } }
          return
        end
        if path == "/v1/novelty"
          text = qparam(ctx, "text") || ""
          v = @lock.synchronize { @mem.novelty(text) }
          respond(ctx, 200) { |j| j.object { j.field "novelty", v } }
          return
        end
        if path == "/v1/similar" && method == "POST"
          similar(ctx)
          return
        end
        if path == "/v1/escalate" && method == "POST"
          escalate(ctx)
          return
        end
        if path == "/v1/kv"
          scan(ctx)
          return
        end
        if path.starts_with?("/v1/kv/")
          key = urldecode(path["/v1/kv/".bytesize..])
          kv(ctx, method, key)
          return
        end
        respond(ctx, 404) { |j| j.object { j.field "error", "no such route" } }
      rescue ex : Exception
        respond(ctx, 500) { |j| j.object { j.field "error", ex.message || ex.class.to_s } }
      end
    end

    private def info_json(j : JSON::Builder)
      j.field "version", Semrember::VERSION
      j.field "uptime_seconds", (Time.utc - @started).total_seconds.to_i64
      j.field "records", @records.size
      j.field "labels", @mem.labels.size
      j.field "d_tfidf", @mem.d_tfidf
      j.field "d_static", @mem.d_static
      j.field "solver", @mem.solver.to_s
      j.field "exact_endpoints", ["GET", "PUT", "DELETE", "EXISTS", "SCAN"]
      j.field "semantic_endpoints", ["SIMILAR", "ESCALATE", "NOVELTY"]
    end

    # ── exact key-value operations ───────────────────────────────────────
    # These never touch the model.  A GET is a hash lookup; the model is
    # only consulted when the caller explicitly asks for a semantic answer.
    private def kv(ctx : HTTP::Server::Context, method : String, key : String)
      case method
      when "GET"
        if key.ends_with?("/exists")
          k = strip_suffix(key, "/exists")
          v = @lock.synchronize { @records.has_key?(k) }
          respond(ctx, 200) { |j| j.object { j.field "exists", v } }
          return
        end
        if key.ends_with?("/ttl")
          k = strip_suffix(key, "/ttl")
          ttl = @lock.synchronize { ttl_of(k) }
          respond(ctx, 200) { |j| j.object { j.field "ttl", ttl.nil? ? JSON::Any.new(nil) : JSON::Any.new(ttl.not_nil!.to_i64) } }
          return
        end
        expired = @lock.synchronize { purge_expired(key); @records[key]? }
        if expired.nil?
          respond(ctx, 404) { |j| j.object { j.field "error", "not found" } }
        else
          respond(ctx, 200) { |j| j.object { j.field "key", key; j.field "value", expired.not_nil! } }
        end
      when "PUT"
        body = read_body(ctx)
        value = jstr(body["value"]?)
        prev = @lock.synchronize do
          purge_expired(key)
          old = @records[key]?
          @records[key] = value
          @expiry.delete(key)
          begin
            @mem.set(key, value)
          rescue ex : Exception
            # The model cannot accept the write (for example the writer state
            # was dropped by load()); the record store still holds the exact
            # value, which is what an exact GET returns.
          end
          old
        end
        respond(ctx, 200) do |j|
          j.object do
            j.field "ok", true
            j.field "previous", prev.nil? ? JSON::Any.new(nil) : JSON::Any.new(prev.not_nil!)
          end
        end
      when "DELETE"
        removed = @lock.synchronize do
          purge_expired(key)
          had = @records.delete(key)
          @expiry.delete(key)
          begin
            @mem.delete(key)
          rescue ex : Exception
          end
          !had.nil?
        end
        respond(ctx, 200) { |j| j.object { j.field "deleted", removed } }
      when "POST"
        if key.ends_with?("/expire")
          k = strip_suffix(key, "/expire")
          body = read_body(ctx)
          secs = jnum(body["seconds"]?)
          ok = @lock.synchronize do
            if @records.has_key?(k) && secs > 0
              @expiry[k] = Time.utc + secs.seconds
              true
            else
              false
            end
          end
          respond(ctx, 200) { |j| j.object { j.field "ok", ok } }
          return
        end
        respond(ctx, 404) { |j| j.object { j.field "error", "no such route" } }
      else
        respond(ctx, 405) { |j| j.object { j.field "error", "method not allowed" } }
      end
    end

    # A deterministic cursor over sorted keys, Redis-style.
    private def scan(ctx : HTTP::Server::Context)
      cursor = (qparam(ctx, "cursor") || "0").to_i
      pattern = qparam(ctx, "match")
      count = (qparam(ctx, "count") || "100").to_i
      count = 100 if count <= 0
      keys = @lock.synchronize do
        purge_all_expired
        @records.keys.sort
      end
      selected = keys[cursor, count]
      selected = selected.select { |k| glob_match?(pattern, k) } if pattern
      next_cursor = cursor + count < keys.size ? (cursor + count).to_s : "0"
      respond(ctx, 200) do |j|
        j.object do
          j.field "cursor", next_cursor
          j.field "keys" do
            j.array { selected.each { |k| j.string k } }
          end
        end
      end
    end

    # ── semantic operations ─────────────────────────────────────────────
    private def similar(ctx : HTTP::Server::Context)
      body = read_body(ctx)
      text = jstr(body["text"]?)
      topk = jnum(body["top_k"]?, 5_i64).to_i32
      min_score = body["min_score"]?
      hits = @lock.synchronize { @mem.similar(text, topk) }
      candidates = [] of Tuple(String, String, Float64)
      hits.each do |(key, score)|
        next if min_score && score.to_f64 < jfloat(min_score)
        value = @lock.synchronize { @records[key]? }
        candidates << {key, value || "", score.to_f64}
      end
      respond(ctx, 200) do |j|
        j.object do
          j.field "candidates" do
            j.array do
              candidates.each do |(key, value, score)|
                j.object do
                  j.field "key", key
                  j.field "value", value
                  j.field "score", score
                end
              end
            end
          end
        end
      end
    end

    # The escalation endpoint: decide, and always say why.  Two independent
    # gates — the scale-free top-2 margin and the out-of-distribution score
    # — catch different failure modes, and the evidence names the stored
    # memories that moved the score.
    private def escalate(ctx : HTTP::Server::Context)
      body = read_body(ctx)
      text = jstr(body["text"]?)
      min_margin = jfloat(body["min_margin"]?, 0.35)
      min_novelty = jfloat(body["min_novelty"]?, 0.55)
      ranked = @lock.synchronize { @mem.query(text, 3) }
      scores = @lock.synchronize { @mem.scores(text) }
      nov = @lock.synchronize { @mem.novelty(text) }
      margin = 0.0_f64
      if scores.size >= 2
        s1 = scores[0].to_f64
        s2 = scores[1].to_f64
        den = s1.abs + s2.abs
        margin = den > 1e-12 ? (s1 - s2) / den : 0.0
      elsif scores.size == 1
        margin = 1.0
      end
      ambiguous = margin < min_margin
      ood = nov > min_novelty
      reason = if ambiguous && ood
                 "both"
               elsif ambiguous
                 "ambiguous"
               elsif ood
                 "out_of_domain"
               else
                 "handled"
               end
      handled = reason == "handled"
      respond(ctx, 200) do |j|
        j.object do
          j.field "decision", handled ? "handle" : "escalate"
          j.field "handled", handled
          j.field "reason", reason
          j.field "margin", margin
          j.field "novelty", nov
          j.field "action", handled && ranked.any? ? ranked[0][0] : JSON::Any.new(nil)
          j.field "candidates" do
            j.array do
              ranked.each do |(label, score)|
                j.object do
                  j.field "label", label
                  j.field "score", score.to_f64
                end
              end
            end
          end
        end
      end
    end

    # ── helpers ─────────────────────────────────────────────────────────
    # Typed accessors for JSON bodies: a missing or mistyped field falls
    # back to the default rather than failing the request.
    private def jstr(v : JSON::Any?, default : String = "") : String
      return default unless v
      v.raw.is_a?(String) ? v.as_s : default
    end

    private def jnum(v : JSON::Any?, default : Int64 = 0_i64) : Int64
      return default unless v
      raw = v.raw
      raw.is_a?(Int64) ? raw : (raw.is_a?(Float64) ? raw.to_i64 : default)
    end

    private def jfloat(v : JSON::Any?, default : Float64 = 0.0) : Float64
      return default unless v
      raw = v.raw
      raw.is_a?(Float64) ? raw : (raw.is_a?(Int64) ? raw.to_f64 : default)
    end

    # Query parameters, tolerating a request with no query string at all.
    private def qparam(ctx : HTTP::Server::Context, name : String) : String?
      ctx.request.query_params[name]?
    end

    private def read_body(ctx : HTTP::Server::Context) : Hash(String, JSON::Any)
      io = ctx.request.body
      return {} of String => JSON::Any unless io
      text = io.gets_to_end
      return {} of String => JSON::Any if text.empty?
      begin
        JSON.parse(text).as_h
      rescue
        {} of String => JSON::Any
      end
    end

    private def respond(ctx : HTTP::Server::Context, status : Int32, &block : JSON::Builder ->)
      ctx.response.status_code = status
      ctx.response.print(JSON.build { |j| block.call(j) })
    end

    # Percent-decoding for path segments.  Deliberately not
    # URI.decode_www_form_component: that turns "+" into a space, which is
    # wrong for a path.
    private def urldecode(text : String) : String
      return text unless text.includes?('%')
      bytes = Bytes.new(text.bytesize)
      size = 0
      i = 0
      src = text.to_slice
      while i < src.size
        b = src[i]
        if b == '%'.ord && i + 2 < src.size
          hex = text.byte_slice(i + 1, 2)
          if hex.matches?(/\A[0-9A-Fa-f]{2}\z/)
            bytes[size] = hex.to_i(16).to_u8
            size += 1
            i += 3
            next
          end
        end
        bytes[size] = b
        size += 1
        i += 1
      end
      String.new(bytes[0, size])
    end

    private def strip_suffix(key : String, suffix : String) : String
      key.byte_slice(0, key.bytesize - suffix.bytesize)
    end

    private def ttl_of(key : String) : Int64?
      deadline = @expiry[key]?
      return nil unless deadline
      remaining = (deadline - Time.utc).total_seconds.to_i64
      remaining < 0 ? nil : remaining
    end

    private def purge_expired(key : String)
      deadline = @expiry[key]?
      return unless deadline
      if deadline <= Time.utc
        @expiry.delete(key)
        @records.delete(key)
      end
    end

    private def purge_all_expired
      now = Time.utc
      @expiry.each { |k, deadline| @records.delete(k) if deadline <= now }
      @expiry.reject! { |_, deadline| deadline <= now }
    end

    # A small glob matcher for SCAN's MATCH argument: * and ? only.
    private def glob_match?(pattern : String?, text : String) : Bool
      return true unless pattern
      return true if pattern == "*"
      glob?(pattern.to_slice, text.to_slice)
    end

    private def glob?(p : Bytes, t : Bytes) : Bool
      return true if p.empty?
      return true if p[0] == '*'.ord && t.empty?
      if p[0] == '*'.ord
        (0..t.size).any? { |i| glob?(p[1..], t[i..]) }
      else
        return false if t.empty?
        return false unless (p[0] == '?'.ord || p[0] == t[0])
        glob?(p[1..], t[1..])
      end
    end
  end
end

# ── entry point ──────────────────────────────────────────────────────────
model = nil.as(String?)
static = nil.as(String?)
train = nil.as(String?)
key_col = "message"
val_col = "action"
port = 7379
host = "127.0.0.1"
args = ARGV.dup
i = 0
while i < args.size
  case args[i]
  when "--model"  then model = args[i + 1]; i += 2
  when "--train"  then train = args[i + 1]; i += 2
  when "--key"    then key_col = args[i + 1]; i += 2
  when "--value"  then val_col = args[i + 1]; i += 2
  when "--static" then static = args[i + 1]; i += 2
  when "--port"   then port = args[i + 1].to_i; i += 2
  when "--host"   then host = args[i + 1]; i += 2
  when "--help", "-h"
    puts "usage: semrember-server [--train corpus.csv --key K --value V | --model model.amem]"
    puts "                      [--static static.bin] [--host H] [--port P]"
    exit 0
  else
    i += 1
  end
end
Semrember::Server.new(model, static, train, key_col, val_col).listen(host, port.to_i32)
