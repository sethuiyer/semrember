# semrember — compressed associative memory
#
# A self-contained Crystal implementation: byte n-gram TF-IDF fused with an
# optional static embedding, a regularized ridge writer, and an exact
# kernel-dual writer that supports replacement and incremental insertion.
#
# The on-disk format is AMEM v3 (see FORMAT.md).  Models written by this
# implementation load in the Python and C implementations, and vice versa.
#
#   mem = Semrember::Memory.new(solver: :dual)
#   mem.fit({"disk full" => "free_disk", "no route to host" => "restart_network"})
#   mem.query("no space left on device")     # => "free_disk"
#
require "set"
require "./codec"
require "./static_encoder"

require "./linalg"
require "./weight_index"

module Semrember
  class UnavailableError < Exception
  end

  VERSION     = "1.0.0"
  MAGIC       = "AMEM"
  FORMAT_VER  =        3
  NGRAM_MIN   =        1
  NGRAM_MAX   =        5
  BETA        =  0.3_f32
  SWEEPS      =       30
  DEFAULT_REG = 0.05_f64
  BLOOM_SEEDS = {2166136261_u32, 2246822519_u32, 3266489917_u32, 668265263_u32}

  # ── hashing / Bloom filter ────────────────────────────────────────────
  module Hashing
    extend self

    def fnv1a(text : String, seed : UInt32) : UInt32
      h = seed
      text.each_byte { |b| h = (h ^ b.to_u32) &* 16777619_u32 }
      h
    end

    # The AMEM header carries a Bloom filter over the vocabulary.  It is not
    # needed to read a model, but it must be reproduced byte-for-byte when
    # writing one, so the layout is fixed here.
    def bloom_bits(terms : Array(String)) : Bytes
      n_bits = 8
      while n_bits < terms.size * 15
        n_bits <<= 1
      end
      bits = Bytes.new(n_bits // 8, 0_u8)
      mask = n_bits - 1
      terms.each do |t|
        BLOOM_SEEDS.each do |seed|
          h = fnv1a(t, seed) & mask
          bits[h >> 3] |= (1 << (h & 7)).to_u8
        end
      end
      bits
    end
  end

  # ── features ──────────────────────────────────────────────────────────
  # A feature vector is the sparse lexical half plus the dense static half.
  struct Features
    getter idx : Array(Int32)
    getter val : Array(Float32)
    getter st : Array(Float32)

    def initialize(@idx : Array(Int32), @val : Array(Float32), @st : Array(Float32))
    end

    # Expand to a dense row of length d_tfidf + d_static.
    def dense(d_tfidf : Int32, d_static : Int32) : Array(Float32)
      row = Array(Float32).new(d_tfidf + d_static, 0_f32)
      @idx.each_with_index { |i, k| row[i] = @val[k] }
      @st.each_with_index { |v, d| row[d_tfidf + d] = v }
      row
    end
  end

  # ── the memory ────────────────────────────────────────────────────────
  class Memory
    getter labels : Array(String)
    getter d_tfidf : Int32
    getter d_static : Int32
    getter min_df : Int32
    getter solver : Symbol
    getter rls_reg : Float64
    getter beta : Float32
    getter sweeps : Int32
    getter stored : Int32
    getter observations_available : Bool
    @vocab : Hash(String, Int32)
    @idf : Array(Float32)
    @bloom : Bytes
    @w : Array(Float32) # T x D, row-major
    @label_idx : Hash(String, Int32)
    @static : StaticEncoder?
    @train_keys : Array(String)
    @train_y : Array(Int32)
    @train_feats : Array(Features)
    @a : Array(Float64)?  # dual inverse, N x N
    @novelty_inverse : Array(Float64)?
    @model_version : Int64
    @index_config : Mips::Config?
    @index : Mips::Index?

    def initialize(@solver : Symbol = :dual, @min_df : Int32 = 1,
                   @sweeps : Int32 = SWEEPS, @beta : Float32 = BETA,
                   @rls_reg : Float64 = DEFAULT_REG,
                   static : StaticEncoder? = nil,
                   index : Mips::Config? = nil)
      raise "solver must be :delta, :rls or :dual" unless {:delta, :rls, :dual}.includes?(@solver)
      raise ArgumentError.new("rls_reg must be finite and > 0") unless @rls_reg.finite? && @rls_reg > 0
      raise ArgumentError.new("min_df must be > 0") if @min_df <= 0
      @static = static
      @d_tfidf = 0
      @d_static = static.try(&.dim) || 0
      @stored = 0
      @vocab = {} of String => Int32
      @idf = [] of Float32
      @bloom = Bytes.empty
      @labels = [] of String
      @label_idx = {} of String => Int32
      @w = [] of Float32
      @train_keys = [] of String
      @train_y = [] of Int32
      @train_feats = [] of Features
      @a = nil
      @novelty_inverse = nil
      @observations_available = true
      @model_version = 0_i64
      @index_config = index
      @index = nil
    end

    def vocab_size : Int32
      @vocab.size.to_i32
    end

    # ── vocabulary ──────────────────────────────────────────────────────
    private def build_vocab(keys : Array(String))
      @vocab.clear
      df = Hash(String, Int32).new(0)
      keys.each do |k|
        seen = Set(String).new
        each_ngram(kb(k)) { |g| seen << g }
        seen.each { |g| df[g] = (df[g]? || 0) + 1 }
      end
      terms = df.keys.select { |t| (df[t]? || 0) >= @min_df }.sort
      terms.each_with_index { |t, i| @vocab[t] = i }
      @d_tfidf = terms.size.to_i32
      n = keys.size
      @idf = terms.map { |t| (Math.log((1.0 + n) / (1.0 + (df[t]? || 0))) + 1.0).to_f32 }
      @bloom = Hashing.bloom_bits(terms)
    end

    # ASCII-only lowercasing, matching bytes.lower() in the reference
    # implementations.
    private def kb(text : String) : String
      src = text.to_slice
      buf = Bytes.new(src.size)
      src.each_with_index do |b, i|
        buf[i] = (b >= 65_u8 && b <= 90_u8 ? b + 32_u8 : b)
      end
      String.new(buf)
    end

    private def each_ngram(text : String, &block : String ->)
      n = text.bytesize
      i = 0
      while i < n
        g = NGRAM_MIN
        while g <= NGRAM_MAX
          break if i + g > n
          yield text.byte_slice(i, g)
          g += 1
        end
        i += 1
      end
    end

    # ── encoding ────────────────────────────────────────────────────────
    # The arithmetic here mirrors the reference implementation closely
    # enough that the two produce bit-identical features: a float32 dot
    # product in numpy is a float64 accumulation rounded to float32, and the
    # joint norm is applied on top of the lexical norm even when there is
    # no static half.
    def encode(text : String) : Features
      lowered = kb(text)
      counts = Hash(String, Int32).new(0)
      each_ngram(lowered) do |g|
        if @vocab.has_key?(g)
          counts[g] = (counts[g]? || 0) + 1
        end
      end
      idx = [] of Int32
      val = [] of Float32
      unless counts.empty?
        pairs = [] of Tuple(Int32, Float32)
        counts.each do |(g, c)|
          i = @vocab[g]
          lf = Math.log(c.to_f64).to_f32
          pairs << {i, (1.0_f32 + lf) * @idf[i]}
        end
        pairs.sort_by! { |(i, _)| i }
        pairs.each do |(i, v)|
          idx << i
          val << v
        end
        lex_norm = Math.sqrt(sumsq(val).to_f64).to_f32 + 1e-9_f32
        val = val.map { |v| v / lex_norm }
      end
      st64 = @static.try(&.encode64(text)) || [] of Float64
      st_sq = 0.0_f64
      st64.each { |v| st_sq += v * v }
      joint = Math.sqrt(sumsq(val).to_f64 + st_sq).to_f32 + 1e-9_f32
      val = val.map { |v| v / joint }
      # A static encoder returns no vector when every token is unknown.  Keep
      # the static feature slot fixed-width so every consumer can index it.
      st = Array(Float32).new(@d_static, 0_f32)
      st64.each_with_index { |v, dd| st[dd] = (v / joint.to_f64).to_f32 if dd < @d_static }
      Features.new(idx, val, st)
    end

    # Sum of squares, accumulated in float64 and rounded to float32 — this is
    # what numpy's float32 dot product does.
    private def sumsq(v : Array(Float32)) : Float32
      s = 0.0_f64
      v.each { |x| s += x.to_f64 * x.to_f64 }
      s.to_f32
    end

    # Dot product over the sparse lexical half.  Both index arrays are sorted
    # by encode, so this is a two-pointer intersection rather than a dense scan.
    private def sparse_dot(a : Features, b : Features) : Float64
      s = 0.0_f64
      ai = 0
      bi = 0
      while ai < a.idx.size && bi < b.idx.size
        av = a.idx[ai]
        bv = b.idx[bi]
        if av == bv
          s += a.val[ai].to_f64 * b.val[bi].to_f64
          ai += 1
          bi += 1
        elsif av < bv
          ai += 1
        else
          bi += 1
        end
      end
      s
    end

    # K = X X^T + λI.  The lexical half is sparse; the static half is dense
    # but low-dimensional (256 in practice).
    private def kernel_matrix(encs : Array(Features)) : Array(Float64)
      n = encs.size
      k = Array(Float64).new(n * n, 0_f64)
      n.times do |i|
        (i + 1).times do |j|
          s = sparse_dot(encs[i], encs[j])
          if @d_static > 0
            a = encs[i].st
            b = encs[j].st
            @d_static.times { |dd| s += a[dd].to_f64 * b[dd].to_f64 }
          end
          k[i * n + j] = s
          k[j * n + i] = s
        end
      end
      n.times { |i| k[i * n + i] += @rls_reg }
      k
    end

    # ── writers ─────────────────────────────────────────────────────────
    def fit(pairs : Hash(String, String), weights : Hash(String, Int32)? = nil)
      @model_version += 1
      @index = nil
      keys = [] of String
      values = [] of String
      pairs.each do |(k, v)|
        n = weights.try(&.[k]?) || 1
        n = 1 if n < 1
        n.times do
          keys << k
          values << v
        end
      end
      @labels = values.uniq
      @label_idx.clear
      @labels.each_with_index { |l, i| @label_idx[l] = i }
      build_vocab(keys)
      encs = keys.map { |k| encode(k) }
      ys = values.map { |v| @label_idx[v] }
      case @solver
      when :dual then dual_rebuild(encs, ys)
      when :rls  then rls_rebuild(encs, ys)
      else            delta_fit(encs, ys)
      end
      @train_keys = keys
      @train_y = ys
      @train_feats = encs
      @stored = keys.size.to_i32
      @novelty_inverse = nil
      @observations_available = true
      self
    end

    def writable? : Bool
      @observations_available && @solver == :dual
    end

    private def d_total : Int32
      @d_tfidf + @d_static
    end

    private def rows : Int32
      @labels.size.to_i32
    end

    private def delta_fit(encs : Array(Features), ys : Array(Int32))
      t = rows
      d = d_total
      @w = Array(Float32).new(t * d, 0_f32)
      @sweeps.times do
        encs.each_with_index do |f, i|
          y = ys[i]
          t.times do |c|
            pred = 0_f32
            f.idx.each_with_index { |ix, k| pred += @w[c * d + ix] * f.val[k] }
            f.st.each_with_index { |v, dd| pred += @w[c * d + @d_tfidf + dd] * v }
            eb = @beta * ((c == y ? 1_f32 : 0_f32) - pred)
            f.idx.each_with_index { |ix, k| @w[c * d + ix] += eb * f.val[k] }
            f.st.each_with_index { |v, dd| @w[c * d + @d_tfidf + dd] += eb * v }
          end
        end
      end
    end

    private def rls_rebuild(encs : Array(Features), ys : Array(Int32))
      d = d_total
      t = rows
      @w = Array(Float32).new(t * d, 0_f32)
      p = Array(Float32).new(d * d, 0_f32)
      d.times { |i| p[i * d + i] = (1.0 / @rls_reg).to_f32 }
      encs.each_with_index do |f, i|
        x = f.dense(@d_tfidf, @d_static)
        px = Array(Float32).new(d, 0_f32)
        d.times do |r|
          s = 0.0_f32
          d.times { |c| s += p[r * d + c] * x[c] }
          px[r] = s
        end
        den = 1.0 + x.zip(px).sum { |a, b| a.to_f64 * b.to_f64 }
        next if den <= 1e-12
        g = px.map { |v| v.to_f64 / den }
        t.times do |c|
          pred = 0.0_f32
          d.times { |dd| pred += @w[c * d + dd] * x[dd] }
          err = (c == ys[i] ? 1.0 : 0.0) - pred.to_f64
          d.times { |dd| @w[c * d + dd] = (@w[c * d + dd].to_f64 + err * g[dd]).to_f32 }
        end
        d.times do |r|
          gr = g[r]
          d.times { |c| p[r * d + c] = (p[r * d + c].to_f64 - gr * px[c].to_f64).to_f32 }
        end
      end
    end

    # Exact ridge solution in the kernel dual: A = (X Xᵀ + λI)⁻¹, W = Y A X.
    #
    # The Gram is built from sparse lexical features plus the dense static
    # half.  The inverse comes from Cholesky, not Gauss-Jordan.  W is formed
    # as (A Y)ᵀ X, so the N×D representer matrix AX is never materialised.
    private def dual_rebuild(encs : Array(Features), ys : Array(Int32))
      d = d_total
      t = rows
      n = encs.size
      @w = Array(Float32).new(t * d, 0_f32)
      @a = nil
      return if n == 0
      k = kernel_matrix(encs)
      a = Linalg.inverse_spd(k, n)
      b = Array(Float64).new(n * t, 0_f64)
      n.times do |i|
        n.times do |j|
          b[i * t + ys[j]] += a[i * n + j]
        end
      end
      # W = (A Y)ᵀ X, accumulated over the sparse feature rows.  For a fixed
      # (label, feature) pair the additions still occur in observation order,
      # so this matches the dense result without materialising N×D.
      w64 = Array(Float64).new(t * d, 0_f64)
      n.times do |i|
        f = encs[i]
        t.times do |c|
          bc = b[i * t + c]
          next if bc == 0.0
          f.idx.each_with_index { |ix, kk| w64[c * d + ix] += bc * f.val[kk].to_f64 }
          f.st.each_with_index { |v, dd| w64[c * d + @d_tfidf + dd] += bc * v.to_f64 }
        end
      end
      @w = w64.map { |v| v.to_f32 }
      @a = a
    end

    # Debug/validation helper: recompute W = Yᵀ(XXᵀ + λI)⁻¹X from the
    # current feature rows with an independent Cholesky solve, and return the
    # relative residual against the incrementally maintained @w.  O(N³), so
    # this is for tests and checkpoints, not the request path.
    def writer_residual : Float64
      n = @train_keys.size
      t = rows
      return 0.0 if n == 0 || t == 0
      d = d_total
      x = Array(Float64).new(n * d, 0_f64)
      @train_feats.each_with_index do |f, i|
        row = f.dense(@d_tfidf, @d_static)
        d.times { |dd| x[i * d + dd] = row[dd].to_f64 }
      end
      k = kernel_matrix(@train_feats)
      a = Linalg.inverse_spd(k, n)
      b = Array(Float64).new(n * t, 0_f64)
      n.times do |i|
        n.times do |j|
          b[i * t + @train_y[j]] += a[i * n + j]
        end
      end
      max_ref = 0.0_f64
      max_diff = 0.0_f64
      t.times do |c|
        d.times do |dd|
          ref = 0.0_f64
          n.times { |i| ref += b[i * t + c] * x[i * d + dd] }
          diff = (@w[c * d + dd].to_f64 - ref).abs
          max_ref = ref.abs if ref.abs > max_ref
          max_diff = diff if diff > max_diff
        end
      end
      max_diff / (max_ref > 1e-12 ? max_ref : 1.0)
    end

    # ── reading ─────────────────────────────────────────────────────────
    def scores(text : String) : Array(Float32)
      scores_features(encode(text))
    end

    private def scores_features(f : Features) : Array(Float32)
      d = d_total
      t = rows
      out = Array(Float32).new(t, 0_f32)
      f.idx.each_with_index do |ix, k|
        t.times { |c| out[c] += @w[c * d + ix] * f.val[k] }
      end
      f.st.each_with_index do |v, dd|
        t.times { |c| out[c] += @w[c * d + @d_tfidf + dd] * v }
      end
      out
    end

    # Exact scores for a subset of labels. The per-label feature order is
    # identical to scores_features, so finalists are bit-for-bit the same.
    private def score_labels(f : Features, labels : Array(Int32)) : Array(Float32)
      d = d_total
      out = Array(Float32).new(labels.size, 0_f32)
      f.idx.each_with_index do |ix, k|
        labels.each_with_index { |c, j| out[j] += @w[c * d + ix] * f.val[k] }
      end
      f.st.each_with_index do |v, dd|
        labels.each_with_index { |j, c| out[c] += @w[j * d + @d_tfidf + dd] * v }
      end
      out
    end

    # ── optional read-side weight index ────────────────────────────────
    # The index is derived from W, never persisted, and never built on a
    # query path. A query uses it only when the published version matches
    # the current model version; otherwise the reference scorer runs.
    def index_version : Int64?
      @index.try(&.version)
    end

    def index_stats : NamedTuple(rank: Int32, nodes: Int32, payload_bytes: Int64, version: Int64)?
      index = @index
      return nil unless index
      {rank: index.rank, nodes: index.node_count,
       payload_bytes: index.payload_bytes, version: index.version}
    end

    def index_ready? : Bool
      idx = @index
      !idx.nil? && idx.version == @model_version
    end

    def index_state : Symbol
      return :disabled unless @index_config
      index_ready? ? :ready : :stale
    end

    def enable_index(config : Mips::Config) : self
      @index_config = config
      @index = nil
      self
    end

    def disable_index : self
      @index_config = nil
      @index = nil
      self
    end

    # Build from an immutable W snapshot, then publish only if the model
    # did not change during construction. Returns false on failure or a
    # version race; callers keep using the full scorer in that case.
    def rebuild_index! : Bool
      config = @index_config
      return false unless config
      return false if @w.empty? || @labels.empty?
      version = @model_version
      snapshot = @w.dup
      begin
        index = Mips::Index.new(snapshot, @labels.size, d_total, config, version: version)
      rescue ArgumentError
        @index = nil
        return false
      end
      if version == @model_version
        @index = index
        true
      else
        false
      end
    end

    def query(text : String, topk : Int32 = 1) : Array(Tuple(String, Float32))
      raise ArgumentError.new("topk must be nonnegative") if topk < 0
      return [] of Tuple(String, Float32) if topk == 0
      f = encode(text)
      index = @index
      if index && index.version == @model_version && topk < @labels.size
        dense = f.dense(@d_tfidf, @d_static)
        result = index.search(dense, topk, expected_version: @model_version) do |ids|
          score_labels(f, ids)
        end
        unless result.fallback?
          return result.hits.map { |hit| {@labels[hit.label], hit.score} }
        end
      end
      s = scores_features(f)
      order = (0...s.size).to_a.sort_by { |i| {-s[i], i} }
      order.first(topk).map { |i| {@labels[i], s[i]} }
    end

    # ── semantic readouts ───────────────────────────────────────────────
    # Nearest stored keys by cosine similarity.  Every stored row is
    # L2-normalised, so the dot product is the cosine.
    def similar(text : String, topk : Int32 = 5) : Array(Tuple(String, Float32))
      raise UnavailableError.new("loaded models have no stored keys; train from the original corpus") unless @observations_available
      raise ArgumentError.new("topk must be nonnegative") if topk < 0
      q = encode(text).dense(@d_tfidf, @d_static)
      scored = [] of Tuple(String, Float32)
      seen = Set(String).new
      @train_feats.each_with_index do |f, i|
        key = @train_keys[i]
        next unless seen.add?(key)

        s = 0.0_f32
        f.idx.each_with_index { |ix, k| s += f.val[k] * q[ix] }
        f.st.each_with_index { |v, dd| s += v * q[@d_tfidf + dd] }
        scored << {key, s}
      end
      scored.sort_by! { |(k, s)| -s }
      scored.first(topk)
    end

    # Out-of-distribution score in [0, 1], measured on the lexical subspace
    # only:  1 - k'(XX' + λI)⁻¹k  with k = X q.
    #
    # The static half is excluded deliberately: a static encoder maps any
    # input, including nonsense, into a plausible point of its embedding
    # space, so including it destroys the separation between "unfamiliar"
    # and "out of distribution".
    def novelty(text : String) : Float64
      raise UnavailableError.new("loaded models have no stored keys; train from the original corpus") unless @observations_available
      return 1.0 if @train_keys.empty?
      dt = @d_tfidf
      return 1.0 if dt == 0
      n = @train_keys.size
      q = encode(text)
      qv = Array(Float64).new(dt, 0_f64)
      q.idx.each_with_index { |i, k| qv[i] = q.val[k].to_f64 }
      nrm = 0.0_f64
      qv.each { |v| nrm += v * v }
      return 1.0 if nrm < 1e-18
      qv = qv.map { |v| v / Math.sqrt(nrm) }
      k = Array(Float64).new(n, 0_f64)
      n.times do |i|
        s = 0.0_f64
        f = @train_feats[i]
        f.idx.each_with_index { |ix, j| s += f.val[j].to_f64 * qv[ix] }
        k[i] = s
      end
      inv = @novelty_inverse ||= build_novelty_inverse
      s = 0.0_f64
      n.times do |i|
        t = 0.0_f64
        n.times { |j| t += inv[i * n + j] * k[j] }
        s += k[i] * t
      end
      (1.0 - s).clamp(0.0, 1.0)
    end

    # The lexical Gram matrix depends on observations, never on the query
    # or their labels. Reuse its inverse until insertion/deletion/refitting.
    private def build_novelty_inverse : Array(Float64)
      # With no static half, the full kernel and the lexical kernel are the
      # same matrix.  Reuse the dual inverse instead of inverting it twice.
      if @d_static == 0
        return @a.not_nil!
      end
      n = @train_feats.size
      gram = Array(Float64).new(n * n, 0_f64)
      n.times do |i|
        (i + 1).times do |j|
          s = sparse_dot(@train_feats[i], @train_feats[j])
          gram[i * n + j] = s
          gram[j * n + i] = s
        end
      end
      n.times { |i| gram[i * n + i] += @rls_reg }
      Linalg.inverse_spd(gram, n)
    end

    def label_count : Int32
      @labels.size.to_i32
    end

    # ── exact writes ────────────────────────────────────────────────────
    # The dual writer supports a *hard* replacement: overwriting a key
    # changes only the target matrix, so the new label is exact and the
    # neighbouring keys move by as little as the algebra allows.  This is
    # what makes a key-value store on top of this model honest — replacing a
    # key replaces its value instead of quietly adding a second observation.
    def set(key : String, value : String) : String?
      raise UnavailableError.new("exact writes require a live dual writer; fit the original corpus first") unless writable?
      @model_version += 1
      @index = nil
      if @train_keys.empty?
        fit({key => value})
        return nil
      end
      previous = previous_value(key)
      vs = value
      unless @label_idx.has_key?(vs)
        @label_idx[vs] = @labels.size
        @labels << vs
        grow_w(@labels.size)
      end
      target = @label_idx[vs]
      hits = [] of Int32
      @train_keys.each_with_index { |k, i| hits << i if k == key }
      if hits.any?
        hits.each { |j| dual_overwrite(j, target) }
        prune_labels
        return previous
      end
      f = encode(key)
      if dual_insert(f, target)
        novelty_insert(f) if @novelty_inverse
        @train_keys << key
        @train_y << target
        @train_feats << f
        @stored = @train_keys.size.to_i32
      else
        # The Schur complement was numerically unsafe.  Preserve the current
        # vocabulary and rebuild the exact dual solution from the retained
        # feature rows plus the new observation.
        @train_keys << key
        @train_y << target
        @train_feats << f
        @stored = @train_keys.size.to_i32
        dual_rebuild(@train_feats, @train_y)
        @novelty_inverse = nil
      end
      previous
    end

    private def previous_value(key : String) : String?
      @train_keys.each_with_index do |k, i|
        next unless k == key
        return @labels[@train_y[i]]
      end
      nil
    end

    private def grow_w(rows : Int32)
      d = d_total
      old = @w
      @w = Array(Float32).new(rows * d, 0_f32)
      rows.times do |c|
        next if c * d + d > old.size
        d.times { |dd| @w[c * d + dd] = old[c * d + dd] }
      end
    end

    # Dead labels must not remain as zero-score candidates after a removal
    # or replacement. Keep the order and weights of the active labels.
    private def prune_labels
      used = @train_y.to_set
      keep = (0...rows).select { |i| used.includes?(i) }
      return if keep.size == rows
      remap = {} of Int32 => Int32
      keep.each_with_index { |old, i| remap[old] = i }
      d = d_total
      @w = keep.flat_map { |i| @w[i * d, d] }
      @labels = keep.map { |i| @labels[i] }
      @label_idx.clear
      @labels.each_with_index { |label, i| @label_idx[label] = i }
      @train_y = @train_y.map { |i| remap[i] }
    end

    # P_i = A_i,: X.  The lexical half is accumulated over sparse rows; the
    # static half is a dense N×d_static product.  No N×D representer cache is
    # needed.
    private def p_row(i : Int32) : Array(Float64)
      n = @train_keys.size
      d = d_total
      a = @a.not_nil!
      p = Array(Float64).new(d, 0_f64)
      n.times do |j|
        aij = a[i * n + j]
        next if aij == 0.0
        f = @train_feats[j]
        f.idx.each_with_index { |ix, k| p[ix] += aij * f.val[k].to_f64 }
        f.st.each_with_index { |v, dd| p[@d_tfidf + dd] += aij * v.to_f64 }
      end
      p
    end

    # Exact rank-1 replacement: W[v_new] += P_j, W[v_old] -= P_j.
    private def dual_overwrite(j : Int32, target : Int32)
      return unless @a
      d = d_total
      old = @train_y[j]
      return if old == target
      r = p_row(j)
      d.times do |dd|
        @w[target * d + dd] += r[dd].to_f32
        @w[old * d + dd] -= r[dd].to_f32
      end
      @train_y[j] = target
    end

    # Exact rank-1 insert: Schur-complement growth of A.  The dense X cache
    # is not needed; k and Xᵀ Ak are accumulated from sparse feature rows.
    private def dual_insert(f : Features, target : Int32) : Bool
      a = @a.not_nil!
      d = d_total
      n = @train_keys.size
      xv = f.dense(@d_tfidf, @d_static).map { |v| v.to_f64 }
      k = Array(Float64).new(n, 0_f64)
      n.times do |i|
        g = @train_feats[i]
        s = sparse_dot(g, f)
        if @d_static > 0
          @d_static.times { |dd| s += g.st[dd].to_f64 * f.st[dd].to_f64 }
        end
        k[i] = s
      end
      ak = Array(Float64).new(n, 0_f64)
      n.times do |i|
        s = 0.0_f64
        n.times { |j| s += a[i * n + j] * k[j] }
        ak[i] = s
      end
      sig = @rls_reg + xv.sum { |v| v * v } - k.zip(ak).sum { |p, q| p * q }
      return false if sig <= 1e-12
      xak = Array(Float64).new(d, 0_f64)
      n.times do |j|
        aj = ak[j]
        next if aj == 0.0
        g = @train_feats[j]
        g.idx.each_with_index { |ix, kk| xak[ix] += aj * g.val[kk].to_f64 }
        g.st.each_with_index { |v, dd| xak[@d_tfidf + dd] += aj * v.to_f64 }
      end
      r = Array(Float64).new(d, 0_f64)
      d.times { |dd| r[dd] = (xv[dd] - xak[dd]) / sig }
      t = rows
      err = Array(Float64).new(t, 0.0_f64)
      t.times do |c|
        s = 0.0_f64
        d.times { |dd| s += @w[c * d + dd] * xv[dd] }
        err[c] = -s
      end
      err[target] += 1.0 if target >= 0 && target < t
      t.times do |c|
        d.times { |dd| @w[c * d + dd] = (@w[c * d + dd].to_f64 + err[c] * r[dd]).to_f32 }
      end
      an = Array(Float64).new((n + 1) * (n + 1), 0_f64)
      n.times do |i|
        n.times { |j| an[i * (n + 1) + j] = a[i * n + j] + ak[i] * ak[j] / sig }
        an[i * (n + 1) + n] = -ak[i] / sig
        an[n * (n + 1) + i] = -ak[i] / sig
      end
      an[n * (n + 1) + n] = 1.0 / sig
      @a = an
      true
    end

    # Keep a cached lexical novelty inverse in sync with a new observation.
    # This is the Schur-complement growth used by dual_insert, but restricted
    # to the lexical feature subspace.
    private def novelty_insert(f : Features) : Nil
      inv = @novelty_inverse
      return unless inv
      n = @train_feats.size
      k = Array(Float64).new(n, 0_f64)
      n.times { |i| k[i] = sparse_dot(@train_feats[i], f) }
      u = Array(Float64).new(n, 0_f64)
      n.times do |i|
        s = 0.0_f64
        n.times { |j| s += inv[i * n + j] * k[j] }
        u[i] = s
      end
      l2 = 0.0_f64
      f.idx.each_with_index { |_, a| v = f.val[a].to_f64; l2 += v * v }
      sig = @rls_reg + l2
      n.times { |i| sig -= k[i] * u[i] }
      if sig <= 1e-12
        @novelty_inverse = nil
        return
      end
      an = Array(Float64).new((n + 1) * (n + 1), 0_f64)
      n.times do |i|
        n.times { |j| an[i * (n + 1) + j] = inv[i * n + j] + u[i] * u[j] / sig }
        an[i * (n + 1) + n] = -u[i] / sig
        an[n * (n + 1) + i] = -u[i] / sig
      end
      an[n * (n + 1) + n] = 1.0 / sig
      @novelty_inverse = an
    end

    def delete(key : String) : Bool
      delete_many({key})
    end

    # Delete one observation in place.  The inverse of the retained kernel is
    # the inverse of the principal submatrix, so this is one outer product:
    #   A' = A_RR - A_Ri A_iR / A_ii
    # W receives the corresponding rank-one correction, and P_i is generated
    # on demand from the sparse feature rows.
    protected def delete_one(i : Int32) : Bool
      n = @train_keys.size
      a = @a.not_nil!
      aii = a[i * n + i]
      return false if aii <= 1e-12

      p_i = p_row(i)
      t = rows
      d = d_total
      coef = Array(Float64).new(t, 0_f64)
      coef[@train_y[i]] += 1.0
      n.times do |r|
        next if r == i
        coef[@train_y[r]] += a[r * n + i] / aii
      end
      t.times do |c|
        d.times do |dd|
          @w[c * d + dd] = (@w[c * d + dd].to_f64 - coef[c] * p_i[dd]).to_f32
        end
      end

      @a = Linalg.delete_principal(a, n, i)
      if inv = @novelty_inverse
        begin
          @novelty_inverse = Linalg.delete_principal(inv, n, i)
        rescue ArgumentError
          @novelty_inverse = nil
        end
      end

      @train_keys.delete_at(i)
      @train_y.delete_at(i)
      @train_feats.delete_at(i)
      @stored = @train_keys.size
      true
    end

    # Numerical fallback for a bad downdate.  This is the old full-rebuild
    # path and should be rare.
    private def rebuild_after_delete(removed : Set(String)) : Bool
      keep = [] of Int32
      @train_keys.each_with_index { |k, i| keep << i unless removed.includes?(k) }
      return false if keep.size == @train_keys.size
      encs = keep.map { |i| @train_feats[i] }
      ys = keep.map { |i| @train_y[i] }
      @train_keys = keep.map { |i| @train_keys[i] }
      @train_y = ys
      @train_feats = encs
      prune_labels
      dual_rebuild(encs, @train_y)
      @stored = keep.size.to_i32
      @novelty_inverse = nil
      true
    end

    # Remove every observation for the given keys.  Each row is deleted with
    # an exact rank-1 downdate; there is no refit unless a downdate becomes
    # numerically unsafe.
    def delete_many(keys : Enumerable(String)) : Bool
      raise UnavailableError.new("exact writes require a live dual writer; fit the original corpus first") unless writable?
      removed = keys.to_set
      return false if removed.empty?

      indices = [] of Int32
      @train_keys.each_with_index { |k, i| indices << i if removed.includes?(k) }
      return false if indices.empty?
      @model_version += 1
      @index = nil
      indices.sort!.reverse!
      indices.each do |i|
        return rebuild_after_delete(removed) unless delete_one(i)
      end
      prune_labels
      @stored = @train_keys.size
      true
    end

    def label(i : Int32) : String
      @labels[i]
    end

    # ── AMEM v3 ─────────────────────────────────────────────────────────
    def save(path : String)
      raise ArgumentError.new("AMEM labels must fit in 65535 bytes") if @labels.any? { |l| l.bytesize > UInt16::MAX }
      File.open(path, "wb") do |f|
        f.write(MAGIC.to_slice)
        terms = @vocab.keys.sort
        f.write(Codec.u32(FORMAT_VER))
        f.write(Codec.u32(terms.size))
        f.write(Codec.u32(@d_static))
        f.write(Codec.u32(@labels.size))
        f.write(Codec.u32(@min_df))
        f.write(Codec.u32(@bloom.size))
        f.write(@bloom)
        terms.each do |t|
          f.write(Codec.u16(t.bytesize))
          f.write(t.to_slice)
          f.write(Codec.f32(@idf[@vocab[t]]))
        end
        @labels.each do |l|
          b = l.to_slice
          f.write(Codec.u16(b.size))
          f.write(b)
        end
        t = rows
        d = d_total
        buf = Bytes.new(t * d * 2)
        t.times do |c|
          d.times do |dd|
            h = F16.from_f32(@w[c * d + dd])
            off = (c * d + dd) * 2
            buf[off] = (h & 0xFF).to_u8
            buf[off + 1] = (h >> 8).to_u8
          end
        end
        f.write(buf)
      end
    end

    # ── loading ────────────────────────────────────────────────────────
    # The loader populates a memory through this protected method so the
    # same invariants apply to a loaded model as to a fitted one.
    protected def adopt(labels : Array(String), vocab : Hash(String, Int32),
                        idf : Array(Float32), bloom : Bytes,
                        d_tfidf : Int32, d_static : Int32,
                        w : Array(Float32))
      @model_version += 1
      @index = nil
      @labels = labels
      @vocab = vocab
      @idf = idf
      @bloom = bloom
      @d_tfidf = d_tfidf
      @d_static = d_static
      @w = w
      @observations_available = false
      @label_idx = {} of String => Int32
      labels.each_with_index { |l, i| @label_idx[l] = i }
    end

    def self.load(path : String, static : StaticEncoder? = nil) : Memory
      data = File.read(path).to_slice
      raise ArgumentError.new("#{path}: truncated AMEM header") if data.size < 28
      raise ArgumentError.new("#{path}: not an AMEM file") unless data[0, 4] == MAGIC.to_slice
      off = 4
      ver = Codec.read_u32(data, off).to_i32; off += 4
      dt = Codec.read_u32(data, off).to_i32; off += 4
      ds = Codec.read_u32(data, off).to_i32; off += 4
      t = Codec.read_u32(data, off).to_i32; off += 4
      min_df = Codec.read_u32(data, off).to_i32; off += 4
      bb = Codec.read_u32(data, off).to_i32; off += 4
      raise "#{path}: AMEM v#{ver}, expected v#{FORMAT_VER}" unless ver == FORMAT_VER
      actual_ds = static.try(&.dim) || 0
      raise ArgumentError.new("#{path}: requires static encoder dimension #{ds}, got #{actual_ds}") unless ds == actual_ds
      minimum = 28_i64 + bb + dt.to_i64 * 6 + t.to_i64 * 2 + t.to_i64 * (dt.to_i64 + ds) * 2
      raise ArgumentError.new("#{path}: truncated AMEM data") if minimum > data.size
      mem = Memory.new(solver: :dual, min_df: min_df, static: static)
      bloom = data[off, bb]
      off += bb
      vocab = {} of String => Int32
      idf = [] of Float32
      dt.times do |i|
        len = Codec.read_u16(data, off).to_i32; off += 2
        term = String.new(data[off, len])
        off += len
        vocab[term] = i
        idf << Codec.read_f32(data, off); off += 4
      end
      labels = [] of String
      t.times do
        len = Codec.read_u16(data, off).to_i32; off += 2
        labels << String.new(data[off, len])
        off += len
      end
      raise ArgumentError.new("#{path}: invalid AMEM weight length") unless data.size.to_i64 - off == t.to_i64 * (dt.to_i64 + ds) * 2
      w = Array(Float32).new(t * (dt + ds), 0_f32)
      (t * (dt + ds)).times do |i|
        w[i] = F16.to_f32(Codec.read_u16(data, off)); off += 2
      end
      mem.adopt(labels, vocab, idf, bloom, dt, ds, w)
      mem
    rescue ex : IndexError | OverflowError
      raise ArgumentError.new("#{path}: invalid or truncated AMEM data")
    end
  end
end
