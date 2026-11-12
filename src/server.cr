# server.cr — the semrember service.
#
#   semrember-server [--model model.amem] [--static static.bin] [--port 7379]
#
# A Redis-shaped HTTP/JSON API over a compressed associative memory.
#
# The split matters and is deliberate:
#
#   exact     GET / PUT / DELETE / EXISTS / SCAN operate on the record
#             store; writes also maintain its semantic index.
#   semantic  SIMILAR / ESCALATE / NOVELTY invoke the model and return
#             ranked candidates, scores and confidence — never a bare
#             answer, because several records may share one.
#
# Redis compatibility is a mental model, not a wire protocol: the transport
# is HTTP+JSON, the command names are borrowed so the concepts transfer, and
# the semantic layer has no Redis equivalent at all.

require "kemal"
require "json"
require "deque"
require "./semrember"
require "./influence"
require "./csv"
require "./encrypted_store"

module Semrember
  class Server
    getter records : Hash(String, String)
    getter expiry : Hash(String, Time)

    @mem : Memory
    @lock : Mutex
    @started : Time
    @store : EncryptedStore?
    @encoder : StaticEncoder?
    @query_buffer : Deque(String)
    @poison_history : Array(NamedTuple(at: Time, key: String, suspicion: Float64, verdict: String))
    @query_buffer_limit : Int32 = 128
    @poison_history_limit : Int32 = 64

    def initialize(model : String?, static : String?, train : String? = nil,
                   key_col : String = "message", val_col : String = "action",
                   data : String? = nil, encryption_key : Bytes? = nil,
                   min_counter : Int64 = 0)
      raise ArgumentError.new("--model and --train are mutually exclusive") if model && train
      raise ArgumentError.new("data and encryption_key must be supplied together") unless data.nil? == encryption_key.nil?
      raise ArgumentError.new("durable mode cannot be combined with --model or --train") if data && (model || train)
      encoder = static.try { |p| StaticEncoder.new(p) }
      @encoder = encoder
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
      @query_buffer = Deque(String).new
      @poison_history = [] of NamedTuple(at: Time, key: String, suspicion: Float64, verdict: String)
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
      if data
        encoder_id = encoder.try { |e| "#{e.tokenizer}:#{e.dim}:#{e.fingerprint}" }
        store = EncryptedStore.new(data, encryption_key.not_nil!, encoder_id, min_counter)
        @store = store
        begin
          snapshot_existed = store.exists?
          @records, @expiry = store.load
          expired = @expiry.select { |_, deadline| deadline <= Time.utc }.keys
          expired.each { |key| @records.delete(key); @expiry.delete(key) }
          @mem = Memory.new(solver: :dual, static: @encoder).fit(@records)
          # Do not rewrite a valid snapshot merely because the process restarted.
          # Persist only a new file, a purge, or newly-recorded encoder metadata.
          if !snapshot_existed || !expired.empty? || store.encoder_upgrade_pending
            store.save(@records, @expiry)
          end
        rescue ex
          store.close
          raise ex
        end
      end
    end

    def close
      @store.try(&.close)
    end

    # Build candidate state first; only publish it after the disk commit succeeds.
    private def commit_change(&block : Hash(String, String), Hash(String, Time) ->)
      records = @records.dup
      expiry = @expiry.dup
      yield records, expiry
      # EXPIRE and same-value PUTs change only deadlines (or nothing at all);
      # reuse the semantic index.  Record changes use exact incremental
      # updates on a private copy instead of refitting the full model.
      candidate = if records == @records
                    @mem
                  else
                    copy = @mem.duplicate_for_influence
                    unless copy.writable?
                      copy = Memory.new(solver: :dual, static: @encoder).fit(records)
                    else
                      @records.each_key { |k| copy.delete(k) unless records.has_key?(k) }
                      records.each do |k, v|
                        next if @records[k]? == v
                        copy.set(k, v)
                      end
                    end
                    copy
                  end
      @store.not_nil!.save(records, expiry)
      @records = records
      @expiry = expiry
      @mem = candidate
    end

    def listen(host : String, port : Int32)
      configure_kemal(host, port)
      puts "semrember #{Semrember::VERSION} listening on http://#{host}:#{port}"
      puts "  exact:    GET PUT DELETE /v1/kv/{key}"
      puts "  semantic: POST /v1/similar  POST /v1/escalate  GET /v1/novelty"
      # Our CLI has already consumed its options. Kemal owns the HTTP
      # lifecycle and graceful shutdown, but must not parse ARGV again.
      Kemal.run(args: nil)
    end

    def configure_kemal(host : String, port : Int32)
      Kemal.config do |config|
        config.app_name = "semrember"
        config.host_binding = host
        config.port = port
        config.env = ENV["KEMAL_ENV"]? || "production"
        config.serve_static = false
        config.show_exceptions = false
      end

      # Keep the memory engine independent of Kemal. The HTTP layer mounts
      # explicit versioned routes and delegates their work to this service.
      api = Kemal::Router.new("/v1")
      {"/ping", "/info", "/novelty", "/kv", "/kv/"}.each do |path|
        api.get(path) { |ctx| handle(ctx); nil }
      end
      api.post("/similar") { |ctx| handle(ctx); nil }
      api.post("/escalate") { |ctx| handle(ctx); nil }
      api.post("/influence") { |ctx| handle(ctx); nil }
      api.post("/compact") { |ctx| handle(ctx); nil }
      {% for method in %w(get put delete post) %}
        api.{{method.id}}("/kv/*key") { |ctx| handle(ctx); nil }
      {% end %}
      api.register_routes

      # Kemal re-renders registered error statuses. Preserve a JSON error
      # the service already produced; framework errors also use JSON.
      {400, 404, 405, 413, 500}.each do |status|
        Kemal.config.add_error_handler(status) do |ctx, _error|
          ctx.response.content_type = "application/json"
          body = ctx.get?("semrember.response")
          body.is_a?(String) ? body : {error: HTTP::Status.new(status).description}.to_json
        end
      end
    end

    # ── routing ─────────────────────────────────────────────────────────
    def handle(ctx : HTTP::Server::Context)
      ctx.response.content_type = "application/json"
      path = ctx.request.path
      method = ctx.request.method == "HEAD" ? "GET" : ctx.request.method
      begin
        @store.try(&.ensure_healthy!)
        if {"/v1/ping", "/v1/info", "/v1/novelty", "/v1/kv"}.includes?(path) && method != "GET"
          respond(ctx, 405) { |j| j.object { j.field "error", "method not allowed" } }
          return
        end
        if path == "/v1/ping"
          respond(ctx, 200) { |j| j.object { j.field "ok", true; j.field "version", Semrember::VERSION } }
          return
        end
        if path == "/v1/info"
          @lock.synchronize do
            purge_all_expired
            respond(ctx, 200) { |j| j.object { info_json(j) } }
          end
          return
        end
        if path == "/v1/novelty"
          text = qparam(ctx, "text") || ""
          v = @lock.synchronize do
            purge_all_expired
            remember_query(text)
            @mem.novelty(text)
          end
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
        if path == "/v1/influence" && method == "POST"
          influence_endpoint(ctx)
          return
        end
        if path == "/v1/compact" && method == "POST"
          compact_endpoint(ctx)
          return
        end
        if path == "/v1/kv"
          scan(ctx)
          return
        end
        if path.starts_with?("/v1/kv/")
          raw_key = path["/v1/kv/".bytesize..]
          operation = nil.as(String?)
          suffixes = method == "GET" ? {"/exists", "/ttl"} : (method == "POST" ? {"/expire"} : Tuple.new)
          suffixes.each do |suffix|
            if raw_key.ends_with?(suffix)
              operation = suffix
              raw_key = strip_suffix(raw_key, suffix)
              break
            end
          end
          key = urldecode(raw_key)
          raise ArgumentError.new("key must be valid UTF-8") unless key.valid_encoding?
          kv(ctx, method, key, operation)
          return
        end
        respond(ctx, 404) { |j| j.object { j.field "error", "no such route" } }
      rescue unavailable : UnavailableError
        respond(ctx, 409) { |j| j.object { j.field "error", unavailable.message } }
      rescue invalid : ArgumentError | JSON::ParseException | TypeCastError | OverflowError
        respond(ctx, 400) { |j| j.object { j.field "error", invalid.message } }
      rescue ex : Exception
        respond(ctx, 500) { |j| j.object { j.field "error", ex.message || ex.class.to_s } }
      end
    end

    private def info_json(j : JSON::Builder)
      j.field "version", Semrember::VERSION
      j.field "uptime_seconds", (Time.utc - @started).total_seconds.to_i64
      j.field "records", @records.size
      j.field "persistence", @store ? "encrypted-snapshot" : "memory"
      j.field "labels", @mem.labels.size
      j.field "d_tfidf", @mem.d_tfidf
      j.field "d_static", @mem.d_static
      j.field "solver", @mem.solver.to_s
      j.field "semantic_updates", @mem.writable?
      j.field "semantic_reads", @mem.observations_available
      j.field "exact_endpoints", ["GET", "PUT", "DELETE", "EXISTS", "SCAN"]
      j.field "semantic_endpoints", ["SIMILAR", "ESCALATE", "NOVELTY"]
    end

    # ── exact key-value operations ───────────────────────────────────────
    # Values come from the dictionary; writes maintain the semantic index.
    private def kv(ctx : HTTP::Server::Context, method : String, key : String, operation : String?)
      case method
      when "GET"
        if operation == "/exists"
          v = @lock.synchronize { purge_expired(key); @records.has_key?(key) }
          respond(ctx, 200) { |j| j.object { j.field "exists", v } }
          return
        end
        if operation == "/ttl"
          ttl = @lock.synchronize { purge_expired(key); ttl_of(key) }
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
        value = required_string(body, "value")
        wants_poison = bool_field(body, "poison_check", false)
        reject_suspicious = bool_field(body, "reject_suspicious", false)
        poison_result = nil.as(PoisonCheckResult?)
        poison_queries = 0
        cumulative_suspicious = 0
        cumulative_suspicion = 0.0_f64
        rejected = false
        prev = @lock.synchronize do
          purge_expired(key)
          old = @records[key]?
          if wants_poison && @mem.observations_available && @mem.stored > 0
            qs = @query_buffer.to_a
            poison_queries = qs.size
            if qs.size >= 3
              poison_result = @mem.poison_check(key, value, qs)
              @poison_history << {at: Time.utc, key: key,
                suspicion: poison_result.not_nil!.suspicion,
                verdict: poison_result.not_nil!.verdict}
              while @poison_history.size > @poison_history_limit
                @poison_history.shift
              end
              recent = @poison_history.last(16)
              cumulative_suspicious = recent.count { |e| e[:verdict] == "suspicious" }
              cumulative_suspicion = recent.sum { |e| e[:suspicion] }
            end
          end
          if reject_suspicious && poison_result &&
             (poison_result.not_nil!.verdict == "suspicious" || cumulative_suspicious >= 3)
            rejected = true
            next old
          end
          if @store
            commit_change { |records, expiry| records[key] = value; expiry.delete(key) }
          else
            @mem.set(key, value) if @mem.writable?
            @records[key] = value
            @expiry.delete(key)
          end
          old
        end
        if rejected
          respond(ctx, 409) do |j|
            j.object do
              j.field "error", "suspicious write rejected"
              j.field "poison_check" do
                j.object do
                  poison_json(j, poison_result.not_nil!, poison_queries, cumulative_suspicious, cumulative_suspicion)
                end
              end
            end
          end
          return
        end
        respond(ctx, 200) do |j|
          j.object do
            j.field "ok", true
            j.field "semantic_updated", @mem.writable?
            j.field "previous", prev.nil? ? JSON::Any.new(nil) : JSON::Any.new(prev.not_nil!)
            if wants_poison
              j.field "poison_check" do
                if result = poison_result
                  j.object do
                    poison_json(j, result, poison_queries, cumulative_suspicious, cumulative_suspicion)
                  end
                else
                  j.object do
                    j.field "verdict", "insufficient_queries"
                    j.field "queries", poison_queries
                  end
                end
              end
            end
          end
        end
      when "DELETE"
        removed = @lock.synchronize do
          purge_expired(key)
          had = @records.has_key?(key)
          if @store
            commit_change { |records, expiry| records.delete(key); expiry.delete(key) } if had
          else
            @mem.delete(key) if @mem.writable?
            @records.delete(key)
            @expiry.delete(key)
          end
          had
        end
        respond(ctx, 200) { |j| j.object { j.field "deleted", removed } }
      when "POST"
        if operation == "/expire"
          body = read_body(ctx)
          secs = jnum(body["seconds"]?)
          ok = @lock.synchronize do
            purge_expired(key)
            if @records.has_key?(key) && secs > 0
              deadline = Time.utc + secs.seconds
              if @store
                commit_change { |_, expiry| expiry[key] = deadline }
              else
                @expiry[key] = deadline
              end
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
      raise ArgumentError.new("cursor must be nonnegative") if cursor < 0
      raise ArgumentError.new("count must be between 1 and 10000") unless 1 <= count <= 10000
      keys = @lock.synchronize do
        purge_all_expired
        @records.keys.sort
      end
      selected = cursor < keys.size ? keys[cursor, {count, keys.size - cursor}.min] : [] of String
      selected = selected.select { |k| glob_match?(pattern, k) } if pattern
      next_cursor = cursor < keys.size && count < keys.size - cursor ? (cursor + count).to_s : "0"
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
      text = required_string(body, "text")
      topk = jnum(body["top_k"]?, 5_i64).to_i32
      min_score = body["min_score"]?
      candidates = @lock.synchronize do
        purge_all_expired
        remember_query(text)
        hits = @mem.similar(text, topk)
        result = [] of Tuple(String, String, Float64)
        hits.each do |(key, score)|
          next if min_score && score.to_f64 < jfloat(min_score)
          if value = @records[key]?
            result << {key, value, score.to_f64}
          end
        end
        result
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
      text = required_string(body, "text")
      min_margin = jfloat(body["min_margin"]?, 0.35)
      min_novelty = jfloat(body["min_novelty"]?, 0.55)
      ranked, nov = @lock.synchronize do
        purge_all_expired
        remember_query(text)
        {@mem.query(text, 3), @mem.novelty(text)}
      end
      margin = 0.0_f64
      if ranked.size >= 2
        s1 = ranked[0][1].to_f64
        s2 = ranked[1][1].to_f64
        den = s1.abs + s2.abs
        margin = den > 1e-12 ? (s1 - s2) / den : 0.0
      elsif ranked.size == 1
        margin = 1.0
      end
      ambiguous = ranked.empty? || margin < min_margin
      ood = ranked.empty? || nov > min_novelty
      reason = if ambiguous && ood
                 "both"
               elsif ambiguous
                 "ambiguous"
               elsif ood
                 "out_of_domain"
               else
                 "handled"
               end
      handled = reason == "handled" && ranked.any?
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

    private def influence_endpoint(ctx : HTTP::Server::Context)
      body = read_body(ctx)
      queries = required_string_array(body, "queries")
      max_examples = jnum(body["max_examples"]?, 3_i64).to_i32
      raise ArgumentError.new("max_examples must be nonnegative") if max_examples < 0
      results = @lock.synchronize do
        purge_all_expired
        raise UnavailableError.new("loaded models have no stored observations") unless @mem.observations_available
        @mem.banzhaf_influence(queries, max_examples)
      end
      respond(ctx, 200) do |j|
        j.object do
          j.field "memories" do
            j.array do
              results.each do |r|
                j.object do
                  j.field "key", r.key
                  j.field "index", r.index
                  j.field "beta", r.banzhaf
                  j.field "pivotal_queries", r.pivotal_queries
                  j.field "mean_influence", r.mean_influence
                  j.field "max_influence", r.max_influence
                  j.field "a_ii", r.a_ii
                  j.field "unstable", r.unstable
                  j.field "examples" do
                    j.array { r.examples.each { |q| j.string q } }
                  end
                end
              end
            end
          end
        end
      end
    end

    private def compact_endpoint(ctx : HTTP::Server::Context)
      body = read_body(ctx)
      queries = required_string_array(body, "queries")
      threshold = jfloat(body["beta_threshold"]?, 0.0)
      coverage = jnum(body["coverage_floor"]?, 0_i64).to_i32
      keep_per_label = jnum(body["keep_per_label"]?, 1_i64).to_i32
      pin = optional_string_array(body, "pin")
      verify = bool_field(body, "verify", true)
      raise ArgumentError.new("coverage_floor must be nonnegative") if coverage < 0
      raise ArgumentError.new("keep_per_label must be nonnegative") if keep_per_label < 0
      result = @lock.synchronize do
        purge_all_expired
        raise UnavailableError.new("loaded models have no stored observations") unless @mem.observations_available
        @mem.compact(queries, beta_threshold: threshold, coverage_floor: coverage,
          pin: pin, keep_per_label: keep_per_label, verify: verify)
      end
      status = result.verified ? 200 : 409
      respond(ctx, status) do |j|
        j.object do
          j.field "ok", result.verified
          j.field "kept", result.kept
          j.field "pruned", result.pruned
          j.field "verified", result.verified
          j.field "changed_queries", result.changed_queries
          j.field "pinned_kept", result.pinned_kept
          j.field "coverage_kept", result.coverage_kept
          j.field "label_kept", result.label_kept
          j.field "pruned_keys", result.pruned_keys.size
          j.field "persistence_note", @store ? "exact records unchanged; compaction is not persisted and restart refits the semantic index" : "exact records unchanged"
        end
      end
    end

    # ── helpers ─────────────────────────────────────────────────────────
    private def required_string(body : Hash(String, JSON::Any), name : String) : String
      value = body[name]?.try(&.as_s?)
      raise ArgumentError.new("#{name} must be a string") unless value
      value
    end

    private def poison_json(j : JSON::Builder, result : PoisonCheckResult,
                            query_count : Int32, cumulative_suspicious : Int32,
                            cumulative_suspicion : Float64)
      j.field "suspicion", result.suspicion
      j.field "pivotal_queries", result.pivotal_queries
      j.field "dissimilar_pivots", result.dissimilar_pivots
      j.field "novelty", result.novelty
      j.field "verdict", result.verdict
      j.field "a_ii", result.a_ii
      j.field "unstable", result.unstable
      j.field "flipped" do
        j.array do
          result.flipped.each do |f|
            j.object do
              j.field "query", f.query
              j.field "old", f.old_label
              j.field "new", f.new_label
              j.field "cosine_to_key", f.cosine
            end
          end
        end
      end
      j.field "cumulative" do
        j.object do
          j.field "query_corpus", query_count
          j.field "suspicious_writes", cumulative_suspicious
          j.field "suspicion_sum", cumulative_suspicion
          j.field "verdict", cumulative_suspicious >= 3 ? "suspicious" : "clean"
        end
      end
    end

    private def required_string_array(body : Hash(String, JSON::Any), name : String) : Array(String)
      value = body[name]?.try(&.as_a?)
      raise ArgumentError.new("#{name} must be an array of strings") unless value
      raise ArgumentError.new("#{name} must not be empty") if value.empty?
      value.map do |v|
        s = v.as_s?
        raise ArgumentError.new("#{name} must be an array of strings") unless s
        s
      end
    end

    private def optional_string_array(body : Hash(String, JSON::Any), name : String) : Array(String)
      value = body[name]?.try(&.as_a?)
      return [] of String unless value
      value.map do |v|
        s = v.as_s?
        raise ArgumentError.new("#{name} must be an array of strings") unless s
        s
      end
    end

    private def bool_field(body : Hash(String, JSON::Any), name : String, default : Bool) : Bool
      value = body[name]?
      return default unless value
      raw = value.raw
      raise ArgumentError.new("#{name} must be a boolean") unless raw.is_a?(Bool)
      raw
    end

    private def remember_query(text : String)
      @query_buffer << text
      while @query_buffer.size > @query_buffer_limit
        @query_buffer.shift
      end
    end

    private def jnum(v : JSON::Any?, default : Int64 = 0_i64) : Int64
      return default unless v
      raw = v.raw
      raise ArgumentError.new("expected integer") unless raw.is_a?(Int64)
      raw
    end

    private def jfloat(v : JSON::Any?, default : Float64 = 0.0) : Float64
      return default unless v
      raw = v.raw
      raise ArgumentError.new("expected number") unless raw.is_a?(Float64) || raw.is_a?(Int64)
      value = raw.to_f64
      raise ArgumentError.new("expected finite number") unless value.finite?
      value
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
      JSON.parse(text).as_h
    end

    private def respond(ctx : HTTP::Server::Context, status : Int32, &block : JSON::Builder ->)
      ctx.response.status_code = status
      body = JSON.build { |j| block.call(j) }
      ctx.set("semrember.response", body)
      ctx.response.print(body)
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
      @store.try(&.ensure_healthy!)
      deadline = @expiry[key]?
      return unless deadline
      if deadline <= Time.utc
        if @store
          commit_change { |records, expiry| records.delete(key); expiry.delete(key) }
        else
          @mem.delete(key) if @mem.writable?
          @expiry.delete(key)
          @records.delete(key)
        end
      end
    end

    private def purge_all_expired
      @store.try(&.ensure_healthy!)
      now = Time.utc
      expired = @expiry.keys.select { |key| @expiry[key] <= now }
      return if expired.empty?

      if @store
        commit_change do |records, expiry|
          expired.each { |key| records.delete(key); expiry.delete(key) }
        end
        return
      end
      @mem.delete_many(expired) if @mem.writable?
      expired.each do |key|
        @expiry.delete(key)
        @records.delete(key)
      end
    end

    # A small glob matcher for SCAN's MATCH argument: * and ? only.
    private def glob_match?(pattern : String?, text : String) : Bool
      return true unless pattern
      return true if pattern == "*"
      glob?(pattern.chars, text.chars)
    end

    private def glob?(p : Array(Char), t : Array(Char)) : Bool
      # Full-string matching with iterative wildcard backtracking. No
      # recursive calls or exponential exploration of multiple stars.
      pi = ti = 0
      star = -1
      retry_at = 0
      while ti < t.size
        if pi < p.size && (p[pi] == '?' || p[pi] == t[ti])
          pi += 1
          ti += 1
        elsif pi < p.size && p[pi] == '*'
          star = pi
          pi += 1
          retry_at = ti
        elsif star >= 0
          retry_at += 1
          ti = retry_at
          pi = star + 1
        else
          return false
        end
      end
      while pi < p.size && p[pi] == '*'
        pi += 1
      end
      pi == p.size
    end
  end
end
