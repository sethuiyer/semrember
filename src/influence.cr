require "./semrember"

module Semrember
  # Per-memory influence over a query corpus.
  #
  # `banzhaf` is the empirical leave-one-out pivotal rate:
  #   Pr_q[ top1(q; M) != top1(q; M \ {i}) ]
  #
  # This is *not* the classical coalition Banzhaf index.  The latter averages
  # over all subsets/coalitions; this quantity measures influence from the
  # full fitted model.  The two names are kept separate deliberately.
  struct InfluenceResult
    getter key : String
    getter index : Int32
    getter banzhaf : Float64
    getter pivotal_queries : Int32
    getter mean_influence : Float64
    getter max_influence : Float64
    getter examples : Array(String)
    getter a_ii : Float64
    getter unstable : Bool

    def initialize(@key, @index, @banzhaf, @pivotal_queries,
                   @mean_influence, @max_influence, @examples,
                   @a_ii, @unstable)
    end
  end

  # A fully expanded numeric explanation of one leave-one-out deletion.
  struct InfluenceTerms
    getter a_ii : Float64
    getter z_i : Float64
    getter b_i : Array(Float64)
    getter c_i : Array(Float64)
    getter base_scores : Array(Float64)
    getter delta : Array(Float64)
    getter fast_scores : Array(Float64)
    getter exact_scores : Array(Float64)
    getter unstable : Bool

    def initialize(@a_ii, @z_i, @b_i, @c_i, @base_scores, @delta,
                   @fast_scores, @exact_scores, @unstable)
    end
  end

  struct PoisonFlip
    getter query : String
    getter old_label : String
    getter new_label : String
    getter cosine : Float64

    def initialize(@query, @old_label, @new_label, @cosine)
    end
  end

  struct PoisonCheckResult
    getter suspicion : Float64
    getter pivotal_queries : Int32
    getter dissimilar_pivots : Int32
    getter novelty : Float64
    getter verdict : String
    getter flipped : Array(PoisonFlip)
    getter a_ii : Float64
    getter unstable : Bool

    def initialize(@suspicion, @pivotal_queries, @dissimilar_pivots,
                   @novelty, @verdict, @flipped, @a_ii, @unstable)
    end
  end

  # Result of a conservative compaction pass.
  struct CompactResult
    getter kept : Int32
    getter pruned : Int32
    getter verified : Bool
    getter changed_queries : Int32
    getter pinned_kept : Int32
    getter coverage_kept : Int32
    getter label_kept : Int32
    getter pruned_keys : Array(String)

    def initialize(@kept, @pruned, @verified, @changed_queries,
                   @pinned_kept, @coverage_kept, @label_kept, @pruned_keys)
    end
  end

  # Optional phase timings for the influence pipeline.
  class BanzhafProfile
    property encode_ms = 0.0
    property kernel_ms = 0.0
    property precompute_ms = 0.0
    property matmul_ms = 0.0
    property scores_ms = 0.0
    property eval_ms = 0.0
    property total_ms = 0.0
  end

  class Memory
    # Exact leave-one-out score vector for observation i:
    #   s(q; M\i) = s(q; M) - z_i(q) * c_i
    # where z(q) = A (X q) and c_i = (A Y)_i / A_ii.
    def leave_one_out_scores(q : String, i : Int32) : Array(Float64)
      a = @a.not_nil!
      n = @train_keys.size
      t = rows
      raise IndexError.new("memory index out of range") unless 0 <= i < n
      f = encode(q)
      k = Array(Float64).new(n, 0_f64)
      n.times do |j|
        g = @train_feats[j]
        s = sparse_dot(g, f)
        if @d_static > 0
          @d_static.times { |dd| s += g.st[dd].to_f64 * f.st[dd].to_f64 }
        end
        k[j] = s
      end
      z = Array(Float64).new(n, 0_f64)
      n.times do |r|
        s = 0.0_f64
        n.times { |j| s += a[r * n + j] * k[j] }
        z[r] = s
      end
      b = influence_ay(a, n, t)
      scores = Array(Float64).new(t, 0_f64)
      t.times do |c|
        s = 0.0_f64
        n.times { |j| s += b[j * t + c] * k[j] }
        scores[c] = s
      end
      diag = a[i * n + i]
      if diag.abs <= influence_diag_floor(a, n)
        return exact_leave_one_out_scores(q, i)
      end
      zi = z[i]
      t.times do |c|
        scores[c] -= zi * b[i * t + c] / diag
      end
      scores
    end

    def leave_one_out_prediction(q : String, i : Int32) : String
      scores = leave_one_out_scores(q, i)
      @labels[argmax_index(scores)]
    end

    # Empirical leave-one-out pivotal power over the supplied queries.
    def banzhaf_influence(queries : Array(String), max_examples : Int32 = 3, profile : BanzhafProfile? = nil) : Array(InfluenceResult)
      n = @train_keys.size
      t = rows
      return [] of InfluenceResult if n == 0 || t == 0
      a = @a.not_nil!
      q_count = queries.size
      t_start = Time.monotonic
      return influence_results_without_queries(n, a) if q_count == 0

      qfeats = queries.map { |q| encode(q) }
      if p = profile
        p.encode_ms = (Time.monotonic - t_start).total_milliseconds
      end
      t_phase = Time.monotonic
      k = influence_query_kernel(qfeats, n, q_count)
      if p = profile
        p.kernel_ms = (Time.monotonic - t_phase).total_milliseconds
      end
      t_phase = Time.monotonic
      b = influence_ay(a, n, t)

      diag = Array(Float64).new(n, 0_f64)
      c = Array(Float64).new(n * t, 0_f64)
      span = Array(Float64).new(n, 0_f64)
      c_norm = Array(Float64).new(n, 0_f64)
      unstable = Array(Bool).new(n, false)
      floor = influence_diag_floor(a, n)
      n.times do |i|
        d = a[i * n + i]
        diag[i] = d
        if d <= floor
          unstable[i] = true
          next
        end
        lo = Float64::INFINITY
        hi = -Float64::INFINITY
        norm2 = 0.0_f64
        t.times do |cc|
          v = b[i * t + cc] / d
          c[i * t + cc] = v
          lo = v if v < lo
          hi = v if v > hi
          norm2 += v * v
        end
        span[i] = hi - lo
        c_norm[i] = Math.sqrt(norm2)
      end
      if p = profile
        p.precompute_ms = (Time.monotonic - t_phase).total_milliseconds
      end

      t_phase = Time.monotonic
      z = Linalg.matmul(a, k, n, n, q_count)
      if p = profile
        p.matmul_ms = (Time.monotonic - t_phase).total_milliseconds
      end
      t_phase = Time.monotonic
      scores = influence_scores(b, k, n, t, q_count)
      if p = profile
        p.scores_ms = (Time.monotonic - t_phase).total_milliseconds
      end
      t_phase = Time.monotonic

      pivotal = Array(Int32).new(n, 0)
      sum_inf = Array(Float64).new(n, 0_f64)
      max_inf = Array(Float64).new(n, 0_f64)
      examples = Array(Array(String)).new(n) { [] of String }

      q_count.times do |qq|
        s = Array(Float64).new(t) { |cc| scores[cc * q_count + qq] }
        winner, margin = winner_and_margin(s)
        n.times do |i|
          zi = z[i * q_count + qq]
          if unstable[i]
            exact = exact_leave_one_out_scores(queries[qq], i)
            inf = norm_difference(s, exact)
            new_winner = argmax_index(exact)
          else
            inf = zi.abs * c_norm[i]
            if zi.abs * span[i] < margin - 1e-12
              sum_inf[i] += inf
              max_inf[i] = inf if inf > max_inf[i]
              next
            end
            new_winner = winner_after_deletion(s, c, i * t, zi, t)
          end
          sum_inf[i] += inf
          max_inf[i] = inf if inf > max_inf[i]
          if new_winner != winner
            pivotal[i] += 1
            if examples[i].size < max_examples
              examples[i] << queries[qq]
            end
          end
        end
      end

      if p = profile
        p.eval_ms = (Time.monotonic - t_phase).total_milliseconds
        p.total_ms = (Time.monotonic - t_start).total_milliseconds
      end
      results = Array(InfluenceResult).new(n)
      n.times do |i|
        results << InfluenceResult.new(
          @train_keys[i], i,
          pivotal[i].to_f64 / q_count,
          pivotal[i],
          sum_inf[i] / q_count,
          max_inf[i],
          examples[i],
          diag[i],
          unstable[i],
        )
      end
      results
    end

    def memory_keys : Array(String)
      @train_keys.dup
    end

    # Expanded numbers for one memory/query pair.  Useful for docs, audits,
    # and debugging the rank-one identity.
    def influence_terms(q : String, i : Int32) : InfluenceTerms
      a = @a.not_nil!
      n = @train_keys.size
      t = rows
      raise IndexError.new("memory index out of range") unless 0 <= i < n
      f = encode(q)
      k = Array(Float64).new(n, 0_f64)
      n.times do |j|
        g = @train_feats[j]
        s = sparse_dot(g, f)
        if @d_static > 0
          @d_static.times { |dd| s += g.st[dd].to_f64 * f.st[dd].to_f64 }
        end
        k[j] = s
      end
      z = Array(Float64).new(n, 0_f64)
      n.times do |r|
        s = 0.0_f64
        n.times { |j| s += a[r * n + j] * k[j] }
        z[r] = s
      end
      b = influence_ay(a, n, t)
      base = Array(Float64).new(t, 0_f64)
      t.times do |c|
        s = 0.0_f64
        n.times { |j| s += b[j * t + c] * k[j] }
        base[c] = s
      end
      diag = a[i * n + i]
      floor = influence_diag_floor(a, n)
      unstable = diag <= floor
      bi = Array(Float64).new(t) { |c| b[i * t + c] }
      if unstable
        exact = exact_leave_one_out_scores(q, i)
        ci = Array(Float64).new(t, 0.0)
        delta = Array(Float64).new(t) { |c| exact[c] - base[c] }
        return InfluenceTerms.new(diag, z[i], bi, ci, base, delta, exact, exact, true)
      end
      ci = Array(Float64).new(t) { |c| bi[c] / diag }
      delta = Array(Float64).new(t) { |c| -z[i] * ci[c] }
      fast = Array(Float64).new(t) { |c| base[c] + delta[c] }
      exact = exact_leave_one_out_scores(q, i)
      InfluenceTerms.new(diag, z[i], bi, ci, base, delta, fast, exact, false)
    end

    # Counterfactual poisoning check for a proposed PUT.
    #
    # The write is applied to a copy.  The new observation's influence is then
    # evaluated against the query corpus.  A legitimate memory should be
    # pivotal mainly for queries similar to itself; a poisoned memory that
    # flips dissimilar queries gets a high suspicion score.
    #
    # This is a model-sensitivity signal, not proof of malicious intent.
    def poison_check(key : String, label : String, queries : Array(String),
                     suspicion_threshold : Float64 = 0.5,
                     novelty_threshold : Float64 = 0.8,
                     dissimilar_cosine : Float64 = 0.3) : PoisonCheckResult
      raise ArgumentError.new("queries must not be empty") if queries.empty?
      if @train_keys.empty?
        return PoisonCheckResult.new(0.0, 0, 0, 1.0, "insufficient_observations",
          [] of PoisonFlip, 0.0, false)
      end

      novelty = novelty(key)
      copy = duplicate_for_influence
      copy.set(key, label)
      index = copy.memory_keys.index(key)
      raise ArgumentError.new("proposed key was not inserted") unless index
      n = copy.stored
      a = copy.@a.not_nil!
      diag = a[index * n + index]
      floor = influence_diag_floor(a, n)
      unstable = diag <= floor
      qfeats = queries.map { |q| copy.encode(q) }
      f_new = copy.@train_feats[index]

      pivotal = 0
      dissimilar = 0
      suspicion_sum = 0.0_f64
      flipped = [] of PoisonFlip
      queries.each_with_index do |q, qq|
        # The true write counterfactual is original model -> proposed model.
        # For a new key that is absence -> insertion; for an existing key it
        # is old label -> new label.
        before = query(q, 1)[0]?.try(&.[0])
        after = copy.query(q, 1)[0]?.try(&.[0])
        next if before == after
        pivotal += 1
        cosine = influence_feature_dot(f_new, qfeats[qq]).clamp(-1.0, 1.0)
        suspicion_sum += 1.0 - cosine
        dissimilar += 1 if cosine < dissimilar_cosine
        if flipped.size < 5
          flipped << PoisonFlip.new(q, before || "", after || "", cosine)
        end
      end

      suspicion = pivotal == 0 ? 0.0 : (suspicion_sum / pivotal).clamp(0.0, 1.0)
      verdict = if pivotal == 0
                  "clean"
                elsif suspicion >= suspicion_threshold &&
                      (novelty >= novelty_threshold || dissimilar >= 2)
                  "suspicious"
                elsif suspicion >= suspicion_threshold * 0.6
                  "uncertain"
                else
                  "clean"
                end
      PoisonCheckResult.new(suspicion, pivotal, dissimilar, novelty, verdict, flipped, diag, unstable)
    end

    private def influence_feature_dot(a : Features, b : Features) : Float64
      s = sparse_dot(a, b)
      if @d_static > 0
        @d_static.times { |dd| s += a.st[dd].to_f64 * b.st[dd].to_f64 }
      end
      s
    end


    # Conservative counterfactual compaction.
    #
    # Removes keys whose every observation has beta <= threshold, unless the
    # key is pinned, covers at least `coverage_floor` queries as the nearest
    # memory, or is needed to keep `keep_per_label` memories for its label.
    #
    # Zero-beta is a leave-one-out property, not a group property.  The
    # default `verify: true` applies the proposed deletion to a copy and
    # aborts if any query decision changes.
    def compact(queries : Array(String), beta_threshold : Float64 = 0.0,
                coverage_floor : Int32 = 0, pin : Array(String) = [] of String,
                keep_per_label : Int32 = 1, verify : Bool = true) : CompactResult
      n = @train_keys.size
      return CompactResult.new(n, 0, true, 0, 0, 0, 0, [] of String) if n == 0 || queries.empty?

      results = banzhaf_influence(queries, 0)
      by_key = Hash(String, Array(InfluenceResult)).new
      results.each { |r| (by_key[r.key] ||= [] of InfluenceResult) << r }

      frequency = Hash(String, Int32).new(0)
      queries.each do |q|
        hit = similar(q, 1)[0]?
        frequency[hit[0]] += 1 if hit
      end

      pinned = pin.to_set
      key_label = Hash(String, String).new
      @train_keys.each_with_index do |key, i|
        key_label[key] ||= @labels[@train_y[i]]
      end

      candidates = Set(String).new
      by_key.each do |key, rows|
        next if pinned.includes?(key)
        next if frequency[key]? && frequency[key] >= coverage_floor && coverage_floor > 0
        next unless rows.all? { |r| r.banzhaf <= beta_threshold }
        candidates << key
      end

      # Keep at least `keep_per_label` keys per label.
      candidates_by_label = Hash(String, Array(InfluenceResult)).new
      by_key.each do |key, rows|
        next unless candidates.includes?(key)
        label = key_label[key]? || ""
        (candidates_by_label[label] ||= [] of InfluenceResult) << rows.first
      end
      kept_by_label = Hash(String, Int32).new(0)
      by_key.each_key do |key|
        next if candidates.includes?(key)
        label = key_label[key]? || ""
        kept_by_label[label] = (kept_by_label[label]? || 0) + 1
      end
      label_kept = 0
      candidates_by_label.each do |label, rows|
        need = keep_per_label - (kept_by_label[label]? || 0)
        next if need <= 0
        rows.sort_by! { |r| -r.mean_influence }
        rows.first(need).each do |r|
          if candidates.delete(r.key)
            label_kept += 1
          end
        end
      end

      if candidates.empty?
        return CompactResult.new(n, 0, true, 0, pinned.size, 0, label_kept, [] of String)
      end

      ordered = candidates.to_a.sort_by { |key| by_key[key].min_of { |r| r.mean_influence } }
      before = queries.map { |q| query(q, 1)[0]?.try(&.[0]) }
      if verify
        trials = [0]
        pruned_keys = compact_batch(ordered, queries, before, trials, max_trials: 64)
      else
        pruned_keys = ordered
        delete_many(pruned_keys)
      end
      after = queries.map { |q| query(q, 1)[0]?.try(&.[0]) }
      changed = before.zip(after).count { |a, b| a != b }
      CompactResult.new(@train_keys.size, pruned_keys.size, changed == 0, changed,
        pinned.size, frequency.count { |_, v| coverage_floor > 0 && v >= coverage_floor },
        label_kept, pruned_keys)
    end

    # Delete a batch only if it preserves the current decisions.  Failed
    # batches are split; this catches the fact that zero beta is not
    # compositional over groups.
    private def compact_batch(keys : Array(String), queries : Array(String),
                              before : Array(String?), trials : Array(Int32),
                              max_trials : Int32) : Array(String)
      return [] of String if keys.empty? || trials[0] >= max_trials
      trials[0] += 1
      if compact_batch_safe?(keys, queries, before)
        delete_many(keys)
        return keys
      end
      return [] of String if keys.size == 1
      mid = keys.size // 2
      left = compact_batch(keys[0...mid], queries, before, trials, max_trials)
      right = compact_batch(keys[mid..], queries, before, trials, max_trials)
      left + right
    end

    private def compact_batch_safe?(keys : Array(String), queries : Array(String),
                                    before : Array(String?)) : Bool
      copy = duplicate_for_influence
      return false unless copy.delete_many(keys)
      queries.each_with_index do |q, idx|
        after = copy.query(q, 1)[0]?.try(&.[0])
        return false unless after == before[idx]
      end
      true
    end

    # Ridge leverage h_i = 1 - λ A_ii.  Useful as a baseline retention score.
    def leverage_scores : Array(Float64)
      a = @a.not_nil!
      n = @train_keys.size
      Array(Float64).new(n) { |i| 1.0 - @rls_reg * a[i * n + i] }
    end

    # Test/debug helper: an independent copy of the full live state.
    def duplicate_for_influence : Memory
      copy = Memory.new(
        solver: @solver, min_df: @min_df, sweeps: @sweeps,
        beta: @beta, rls_reg: @rls_reg, static: @static,
      )
      copy.copy_state_from(self)
      copy
    end

    protected def copy_state_from(other : Memory)
      @labels = other.@labels.dup
      @label_idx = other.@label_idx.dup
      @vocab = other.@vocab.dup
      @idf = other.@idf.dup
      @bloom = other.@bloom.dup
      @w = other.@w.dup
      @train_keys = other.@train_keys.dup
      @train_y = other.@train_y.dup
      @train_feats = other.@train_feats.dup
      @a = other.@a.try(&.dup)
      @novelty_inverse = other.@novelty_inverse.try(&.dup)
      @stored = other.@stored
      @d_tfidf = other.@d_tfidf
      @d_static = other.@d_static
      @observations_available = other.@observations_available
      @index = nil
      @index_config = other.@index_config
      @model_version = other.@model_version &+ 1
    end

    # Float64 score vector computed from A and the sparse features.  This is
    # the reference for the leave-one-out tests; it does not use W.
    def scores64(q : String) : Array(Float64)
      a = @a.not_nil!
      n = @train_keys.size
      t = rows
      f = encode(q)
      k = Array(Float64).new(n, 0_f64)
      n.times do |j|
        g = @train_feats[j]
        s = sparse_dot(g, f)
        if @d_static > 0
          @d_static.times { |dd| s += g.st[dd].to_f64 * f.st[dd].to_f64 }
        end
        k[j] = s
      end
      b = influence_ay(a, n, t)
      scores = Array(Float64).new(t, 0_f64)
      t.times do |c|
        s = 0.0_f64
        n.times { |j| s += b[j * t + c] * k[j] }
        scores[c] = s
      end
      scores
    end

    # Correctness oracle: delete observation i from a copy and score q.
    def exact_leave_one_out_scores(q : String, i : Int32) : Array(Float64)
      copy = duplicate_for_influence
      unless copy.delete_one(i)
        copy.delete(@train_keys[i])
      end
      copy.scores64(q)
    end

    private def influence_query_kernel(qfeats : Array(Features), n : Int32, q_count : Int32) : Array(Float64)
      k = Array(Float64).new(n * q_count, 0_f64)
      n.times do |i|
        f = @train_feats[i]
        q_count.times do |qq|
          g = qfeats[qq]
          s = sparse_dot(f, g)
          if @d_static > 0
            @d_static.times { |dd| s += f.st[dd].to_f64 * g.st[dd].to_f64 }
          end
          k[i * q_count + qq] = s
        end
      end
      k
    end

    private def influence_ay(a : Array(Float64), n : Int32, t : Int32) : Array(Float64)
      b = Array(Float64).new(n * t, 0_f64)
      n.times do |i|
        n.times do |j|
          b[i * t + @train_y[j]] += a[i * n + j]
        end
      end
      b
    end

    private def influence_scores(b : Array(Float64), k : Array(Float64), n : Int32, t : Int32, q_count : Int32) : Array(Float64)
      scores = Array(Float64).new(t * q_count, 0_f64)
      q_count.times do |qq|
        t.times do |cc|
          s = 0.0_f64
          n.times { |i| s += b[i * t + cc] * k[i * q_count + qq] }
          scores[cc * q_count + qq] = s
        end
      end
      scores
    end

    private def influence_diag_floor(a : Array(Float64), n : Int32) : Float64
      max_diag = 0.0_f64
      n.times do |i|
        v = a[i * n + i]
        max_diag = v if v > max_diag
      end
      1e-12 * (max_diag > 0.0 ? max_diag : 1.0)
    end

    private def influence_results_without_queries(n : Int32, a : Array(Float64)) : Array(InfluenceResult)
      floor = influence_diag_floor(a, n)
      Array(InfluenceResult).new(n) do |i|
        d = a[i * n + i]
        InfluenceResult.new(@train_keys[i], i, 0.0, 0, 0.0, 0.0,
          [] of String, d, d <= floor)
      end
    end

    private def winner_and_margin(scores : Array(Float64)) : Tuple(Int32, Float64)
      t = scores.size
      return {0, Float64::INFINITY} if t <= 1
      winner = 0
      best = scores[0]
      second = -Float64::INFINITY
      (1...t).each do |c|
        v = scores[c]
        if v > best
          second = best
          best = v
          winner = c
        elsif v > second
          second = v
        end
      end
      {winner, best - second}
    end

    private def argmax_index(scores : Array(Float64)) : Int32
      winner = 0
      best = scores[0]
      (1...scores.size).each do |c|
        if scores[c] > best
          best = scores[c]
          winner = c
        end
      end
      winner
    end

    private def winner_after_deletion(scores : Array(Float64), c : Array(Float64), off : Int32, z : Float64, t : Int32) : Int32
      winner = 0
      best = scores[0] - z * c[off]
      (1...t).each do |cc|
        v = scores[cc] - z * c[off + cc]
        if v > best
          best = v
          winner = cc
        end
      end
      winner
    end

    private def norm_difference(a : Array(Float64), b : Array(Float64)) : Float64
      s = 0.0_f64
      a.each_with_index do |v, i|
        d = v - b[i]
        s += d * d
      end
      Math.sqrt(s)
    end
  end
end
