# maxsat.cr — an independent Crystal implementation of the NuWLS-style
# weighted MaxSAT local search used by SUTRA.
#
# This is a port of the *math*, not a wrapper around the C solver:
#
#   * score[v] is the change in satisfied weight if v flips;
#   * a good-variable stack drives the descent;
#   * BMS sampling picks a near-best improving variable in O(1);
#   * additive capped clause weights escape local minima;
#   * a continuous Adam relaxation (hard log barrier + soft hinge) seeds
#     the discrete repair.
#
# No optimality claim. The solver returns the best assignment it found
# and the cost is always recounted from that assignment.
#
# Usage:
#   crystal run src/maxsat_cli.cr -- instance.wcnf --time-budget-seconds 30
#
require "json"

module Semrember
  module MaxSat
    VERSION = "0.1.0"
    HARD_W  = 1.0e9
    EPS     = 1.0e-9

    # ── formula ────────────────────────────────────────────────────────
    # Clauses are stored in CSR form. Literals are signed, 1-based DIMACS
    # integers: +v means "variable v is true", -v means "v is false".
    class Formula
      getter nv : Int32
      getter offset : Array(Int32)
      getter lits : Array(Int32)
      getter weight : Array(Float64)
      getter hard : Array(UInt8)

      def initialize(@nv : Int32, @offset : Array(Int32),
                     @lits : Array(Int32), @weight : Array(Float64),
                     @hard : Array(UInt8))
      end

      def nc : Int32
        @hard.size
      end
    end

    # Lenient DIMACS/WCNF reader. Supports `p cnf`, `p wcnf ... top`,
    # `h`-prefixed hard clauses, integer soft weights, comments, and
    # clauses split across lines. A soft weight of 0 is hard, matching
    # SUTRA's convention.
    module Wcnf
      extend self

      def parse(text : String) : Formula
        nv = 0
        top = -1_i64
        format = ""
        offset = [0_i32]
        lits = [] of Int32
        weights = [] of Float64
        hards = [] of UInt8
        current = [] of Int32
        pending_weight = 1.0
        pending_hard = 1_u8

        text.each_line do |raw|
          line = raw.strip
          next if line.empty?
          next if line.starts_with?('c')
          if line.starts_with?('p')
            parts = line.split
            raise ArgumentError.new("malformed DIMACS header") if parts.size < 4
            format = parts[1]
            nv = parts[2].to_i
            top = parts.size >= 5 ? parts[4].to_i64 : -1_i64
            next
          end

          hard_line = false
          if line.starts_with?('h') || line.starts_with?('H')
            hard_line = true
            line = line[1..].strip
          end
          tokens = line.split
          if current.empty?
            if format == "wcnf" && !hard_line
              word = tokens.shift?
              raise ArgumentError.new("missing WCNF weight") unless word
              wi = word.to_i64?
              raise ArgumentError.new("invalid WCNF weight #{word}") unless wi
              pending_weight = wi.to_f64
              pending_hard = (wi <= 0 || (top > 0 && wi >= top)) ? 1_u8 : 0_u8
            else
              pending_weight = 1.0
              pending_hard = 1_u8
            end
          end

          tokens.each do |tok|
            lit = tok.to_i?
            raise ArgumentError.new("invalid literal #{tok}") unless lit
            if lit == 0
              offset << (offset.last + current.size).to_i32
              current.each { |x| lits << x }
              weights << pending_weight
              hards << pending_hard
              current.clear
              pending_weight = 1.0
              pending_hard = 1_u8
            else
              current << lit
              nv = lit.abs if lit.abs > nv
            end
          end
        end

        raise ArgumentError.new("unterminated clause") unless current.empty?
        raise ArgumentError.new("missing DIMACS header") if format.empty?
        Formula.new(nv, offset, lits, weights, hards)
      end
    end

    # ── deterministic RNG (xorshift64) ─────────────────────────────────
    class Rng
      def initialize(@state : UInt64)
        @state = 0x9E3779B97F4A7C15_u64 if @state == 0
      end

      def next_u64 : UInt64
        x = @state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        @state = x
        x
      end

      def next_double : Float64
        (next_u64 >> 11).to_f64 * (1.0 / 9007199254740992.0)
      end

      def next_int(n : Int32) : Int32
        (next_u64 % n.to_u64).to_i32
      end
    end

    # ── NuWLS search state ─────────────────────────────────────────────
    # score[v] = weight gained by flipping v:
    #   +w for every violated clause containing v
    #   -w for every satisfied clause where v is the only true literal.
    class SearchState
      getter f : Formula
      getter score : Array(Float64)
      getter nh : Int32
      getter ns : Int32
      getter good_n : Int32

      @occ_off : Array(Int32)
      @occ : Array(Int32)
      @occ_sign : Array(Int8)
      @sat_count : Array(Int32)
      @sat_var : Array(Int32)
      @w : Array(Float64)
      @tw : Array(Float64)
      @good : Array(Int32)
      @good_pos : Array(Int32)
      @stamp : Array(Int64)
      @hunsat : Array(Int32)
      @hpos : Array(Int32)
      @sunsat : Array(Int32)
      @spos : Array(Int32)
      @large : Array(Int32)
      @large_pos : Array(Int32)
      @nlarge : Int32
      @rng : Rng

      def initialize(@f : Formula, asn : Array(UInt8), seed : UInt64)
        nv = @f.nv
        nc = @f.nc
        @nh = 0
        @ns = 0
        @nlarge = 0
        @good_n = 0
        @rng = Rng.new(seed)

        # occurrence lists: variable -> clauses containing it, with sign
        @occ_off = Array(Int32).new(nv + 1, 0)
        @f.lits.each do |lit|
          v = lit.abs - 1
          @occ_off[v + 1] += 1
        end
        1.upto(nv) { |v| @occ_off[v] += @occ_off[v - 1] }
        total_occ = @occ_off[nv]
        @occ = Array(Int32).new(total_occ, 0)
        @occ_sign = Array(Int8).new(total_occ, 0_i8)
        fill = Array(Int32).new(nv, 0)
        @f.nc.times do |c|
          (@f.offset[c]...@f.offset[c + 1]).each do |j|
            lit = @f.lits[j]
            v = lit.abs - 1
            pos = @occ_off[v] + fill[v]
            @occ[pos] = c
            @occ_sign[pos] = lit > 0 ? 1_i8 : -1_i8
            fill[v] += 1
          end
        end

        # tuned weights: soft average 1000, hard dominates lexicographically
        nsoft = 0
        total = 0.0
        nc.times do |c|
          if @f.hard[c] == 0
            nsoft += 1
            total += @f.weight[c]
          end
        end
        avg = nsoft > 0 ? total / nsoft.to_f64 : 1.0
        coe = nsoft > 0 ? 1000.0 / avg : 1.0
        @w = Array(Float64).new(nc, 0.0)
        @tw = Array(Float64).new(nc, 0.0)
        nc.times do |c|
          if @f.hard[c] == 1
            @tw[c] = HARD_W
            @w[c] = HARD_W
          elsif nsoft > 0
            @tw[c] = @f.weight[c] * coe
            @w[c] = @tw[c]
          else
            @tw[c] = 1000.0
            @w[c] = 1000.0
          end
        end

        # truth counts, unique true variable, violated-clause stacks
        @sat_count = Array(Int32).new(nc, 0)
        @sat_var = Array(Int32).new(nc, -1)
        @hunsat = Array(Int32).new(nc, 0)
        @hpos = Array(Int32).new(nc, -1)
        @sunsat = Array(Int32).new(nc, 0)
        @spos = Array(Int32).new(nc, -1)
        @large = Array(Int32).new(nc, 0)
        @large_pos = Array(Int32).new(nc, -1)
        nc.times do |c|
          cnt = 0
          sv = -1
          (@f.offset[c]...@f.offset[c + 1]).each do |j|
            lit = @f.lits[j]
            v = lit.abs - 1
            if (lit > 0) == (asn[v] != 0)
              cnt += 1
              sv = v
            end
          end
          @sat_count[c] = cnt
          @sat_var[c] = sv
          unsat_add(c) if cnt == 0
        end

        @score = Array(Float64).new(nv, 0.0)
        @good = Array(Int32).new(nv, 0)
        @good_pos = Array(Int32).new(nv, -1)
        @stamp = Array(Int64).new(nv, 0_i64)
        rebuild_scores

        nc.times do |c|
          if @w[c] > 3.0 && @f.hard[c] == 0
            @large_pos[c] = @nlarge
            @large[@nlarge] = c
            @nlarge += 1
          end
        end
      end

      # Recompute score[] from the definition. Used by tests and
      # after weight movements; the search itself updates incrementally.
      def rebuild_scores : Nil
        @score.fill(0.0)
        @f.nc.times do |c|
          if @sat_count[c] == 0
            (@f.offset[c]...@f.offset[c + 1]).each do |j|
              v = @f.lits[j].abs - 1
              @score[v] += @w[c]
            end
          elsif @sat_count[c] == 1
            @score[@sat_var[c]] -= @w[c]
          end
        end
        @good_n = 0
        @f.nv.times { |v| @good_pos[v] = -1 }
        @f.nv.times { |v| good_add(v) }
      end

      private def good_add(v : Int32) : Nil
        return if v < 0
        if @score[v] > 0 && @good_pos[v] < 0
          @good_pos[v] = @good_n
          @good[@good_n] = v
          @good_n += 1
        end
      end

      private def good_del(v : Int32) : Nil
        return if v < 0
        if @good_pos[v] >= 0
          @good_n -= 1
          last = @good[@good_n]
          @good[@good_pos[v]] = last
          @good_pos[last] = @good_pos[v]
          @good_pos[v] = -1
        end
      end

      private def unsat_add(c : Int32) : Nil
        if @f.hard[c] == 1
          @hpos[c] = @nh
          @hunsat[@nh] = c
          @nh += 1
        else
          @spos[c] = @ns
          @sunsat[@ns] = c
          @ns += 1
        end
      end

      private def unsat_del(c : Int32) : Nil
        if @f.hard[c] == 1
          idx = @hpos[c]
          return if idx < 0
          @nh -= 1
          last = @hunsat[@nh]
          @hunsat[idx] = last
          @hpos[last] = idx
          @hpos[c] = -1
        else
          idx = @spos[c]
          return if idx < 0
          @ns -= 1
          last = @sunsat[@ns]
          @sunsat[idx] = last
          @spos[last] = idx
          @spos[c] = -1
        end
      end

      # One NuWLS flip. The four clause transitions (0->1, 1->2, 2->1,
      # 1->0) update every affected score; the final score[v] = -score[v]
      # is the invariant that makes flipping back undo the gain.
      def flip(asn : Array(UInt8), v : Int32, step : Int64) : Nil
        org = @score[v]
        asn[v] = asn[v] == 0 ? 1_u8 : 0_u8
        k = @occ_off[v]
        while k < @occ_off[v + 1]
          c = @occ[k]
          sign = @occ_sign[k]
          now_true = (sign > 0) == (asn[v] != 0)
          if now_true
            @sat_count[c] += 1
            if @sat_count[c] == 2
              @score[@sat_var[c]] += @w[c]
              good_add(@sat_var[c])
            elsif @sat_count[c] == 1
              @sat_var[c] = v
              j = @f.offset[c]
              while j < @f.offset[c + 1]
                u = @f.lits[j].abs - 1
                @score[u] -= @w[c]
                good_del(u) if @score[u] <= 0
                j += 1
              end
              unsat_del(c)
            end
          else
            @sat_count[c] -= 1
            if @sat_count[c] == 1
              j = @f.offset[c]
              while j < @f.offset[c + 1]
                lit = @f.lits[j]
                u = lit.abs - 1
                if (lit > 0) == (asn[u] != 0)
                  @score[u] -= @w[c]
                  good_del(u) if @score[u] <= 0
                  @sat_var[c] = u
                  break
                end
                j += 1
              end
            elsif @sat_count[c] == 0
              j = @f.offset[c]
              while j < @f.offset[c + 1]
                u = @f.lits[j].abs - 1
                @score[u] += @w[c]
                good_add(u)
                j += 1
              end
              unsat_add(c)
            end
          end
          k += 1
        end
        @score[v] = -org
        good_add(v) if @score[v] > 0
        good_del(v) if @score[v] <= 0
        @stamp[v] = step
      end

      # Additive capped weighting: violated soft clauses get +3, and a
      # rare smoothing pass takes 3 back from satisfied heavy clauses.
      def update_weights : Nil
        if @rng.next_double < 1.0e-3 && @nlarge > 0
          i = 0
          while i < @nlarge
            c = @large[i]
            if @sat_count[c] > 0
              @w[c] -= 3.0
              if @w[c] <= 3.0
                @nlarge -= 1
                @large[i] = @large[@nlarge]
                @large_pos[c] = -1
                next
              end
              if @sat_count[c] == 1
                @score[@sat_var[c]] += 3.0
                good_add(@sat_var[c])
              end
            end
            i += 1
          end
          return
        end

        i = 0
        while i < @ns
          c = @sunsat[i]
          if @w[c] < @tw[c] + 10.0
            @w[c] += 3.0
            if @w[c] > 3.0 && @large_pos[c] < 0
              @large_pos[c] = @nlarge
              @large[@nlarge] = c
              @nlarge += 1
            end
            j = @f.offset[c]
            while j < @f.offset[c + 1]
              u = @f.lits[j].abs - 1
              @score[u] += 3.0
              good_add(u)
              j += 1
            end
          end
          i += 1
        end
      end

      # Case A: an improving variable exists -> BMS sample the good stack.
      # Case B: local minimum -> reweight, then random-walk a violated clause.
      def pick : Int32
        if @good_n > 0
          if @rng.next_double < 0.036
            return @good[@rng.next_int(@good_n)]
          end
          best = @good[@rng.next_int(@good_n)]
          21.times do
            u = @good[@rng.next_int(@good_n)]
            if @score[u] > @score[best] ||
               (@score[u] == @score[best] && @stamp[u] < @stamp[best])
              best = u
            end
          end
          return best
        end

        update_weights
        c = if @nh > 0
              @hunsat[@rng.next_int(@nh)]
            elsif @ns > 0
              @sunsat[@rng.next_int(@ns)]
            else
              return -1
            end
        len = @f.offset[c + 1] - @f.offset[c]
        if @rng.next_double < 0.48
          lit = @f.lits[@f.offset[c] + @rng.next_int(len)]
          return lit.abs - 1
        end
        best = -1
        j = @f.offset[c]
        while j < @f.offset[c + 1]
          u = @f.lits[j].abs - 1
          if best < 0 || @score[u] > @score[best] ||
             (@score[u] == @score[best] && @stamp[u] < @stamp[best])
            best = u
          end
          j += 1
        end
        best
      end

      # Independent recount: hard violations, then original soft cost.
      def cost(asn : Array(UInt8)) : Tuple(Int64, Float64)
        hard = 0_i64
        soft = 0.0
        @f.nc.times do |c|
          sat = false
          j = @f.offset[c]
          while j < @f.offset[c + 1]
            lit = @f.lits[j]
            v = lit.abs - 1
            if (lit > 0) == (asn[v] != 0)
              sat = true
              break
            end
            j += 1
          end
          unless sat
            if @f.hard[c] == 1
              hard += 1
            else
              soft += @f.weight[c]
            end
          end
        end
        {hard, soft}
      end
    end

    # ── discrete NuWLS repair ──────────────────────────────────────────
    module NuWLS
      extend self

      # Run the flip loop until the budget/time is exhausted. Returns the
      # number of flips used, whether every clause was satisfied, and the
      # best assignment seen (recounted, not bookkeeping).
      def repair(f : Formula, asn : Array(UInt8), flips_budget : Int64,
                 time_budget : Float64, t0 : Time::Span,
                 seed : UInt64) : Tuple(Int64, Bool, Array(UInt8))
        state = SearchState.new(f, asn, seed)
        best = asn.dup
        best_h, best_s = state.cost(asn)
        used = 0_i64
        converged = false

        while used < flips_budget
          if state.nh == 0 && state.ns == 0
            converged = true
            break
          end
          if (used & 1023_i64) == 0 && time_budget > 0 &&
             (Time.monotonic - t0).total_seconds > time_budget
            break
          end
          v = state.pick
          break if v < 0
          state.flip(asn, v, used + 1)
          used += 1
          # The lexicographic objective makes the first hard-feasible
          # assignment important; do not wait 256 flips to record it.
          if state.nh == 0 && best_h > 0
            h, s = state.cost(asn)
            if h < best_h || (h == best_h && s < best_s - 1.0e-12)
              best_h = h
              best_s = s
              best = asn.dup
            end
          end
          if (used & 255_i64) == 0
            h, s = state.cost(asn)
            if h < best_h || (h == best_h && s < best_s - 1.0e-12)
              best_h = h
              best_s = s
              best = asn.dup
            end
          end
        end

        if state.nh == 0 && state.ns == 0
          converged = true
          best = asn.dup
        end
        {used, converged, best}
      end
    end

    # ── continuous relaxation (Adam warm start) ────────────────────────
    module Continuous
      extend self

      BETA1 = 0.9
      BETA2 = 0.999

      # One gradient step:
      #   hard:  -log(1 - P(clause false)) log barrier
      #   soft:  squared hinge / product penalty, weighted
      # plus an annealed crispness regularizer and Adam.
      def gradient_step(f : Formula, x : Array(Float64), g : Array(Float64),
                        m : Array(Float64), v : Array(Float64),
                        step : Int32, opts : Options) : Nil
        nv = f.nv
        g.fill(0.0)

        f.nc.times do |c|
          off0 = f.offset[c]
          off1 = f.offset[c + 1]
          width = off1 - off0
          next if width == 0

          if f.hard[c] == 1
            viol = 1.0
            j = off0
            while j < off1
              lit = f.lits[j]
              var = lit.abs
              viol *= lit > 0 ? 1.0 - x[var] : x[var]
              j += 1
            end
            slack = 1.0 - viol
            slack = EPS if slack < EPS
            coef = 64.0 / slack
            j = off0
            while j < off1
              lit = f.lits[j]
              var = lit.abs
              fp = lit > 0 ? 1.0 - x[var] : x[var]
              fp = EPS if fp < EPS
              g[var] += coef * (viol / fp) * (lit > 0 ? -1.0 : 1.0)
              j += 1
            end
          elsif width == 2
            viol = 1.0
            j = off0
            while j < off1
              lit = f.lits[j]
              var = lit.abs
              viol *= lit > 0 ? 1.0 - x[var] : x[var]
              j += 1
            end
            coef = 2.0 * f.weight[c] * viol
            j = off0
            while j < off1
              lit = f.lits[j]
              var = lit.abs
              fp = lit > 0 ? 1.0 - x[var] : x[var]
              fp = EPS if fp < EPS
              g[var] += coef * (viol / fp) * (lit > 0 ? -1.0 : 1.0)
              j += 1
            end
          else
            sum_signed = 0.0
            j = off0
            while j < off1
              lit = f.lits[j]
              var = lit.abs
              sum_signed += lit > 0 ? x[var] : 1.0 - x[var]
              j += 1
            end
            u = 1.0 - sum_signed
            tau = opts.tau > 1.0e-3 ? opts.tau : 1.0e-3
            vv = u / tau
            soft_viol = vv > 0 ? u + tau * Math.log1p(Math.exp(-vv)) : tau * Math.log1p(Math.exp(vv))
            sig = vv >= 0 ? 1.0 / (1.0 + Math.exp(-vv)) : Math.exp(vv) / (1.0 + Math.exp(vv))
            coef = 2.0 * f.weight[c] * soft_viol
            j = off0
            while j < off1
              lit = f.lits[j]
              var = lit.abs
              g[var] += coef * (lit > 0 ? -sig : sig)
              j += 1
            end
          end
        end

        steps = opts.steps > 0 ? opts.steps : 1
        anneal = (step.to_f64 / steps.to_f64) ** 2.0
        mu = 0.05 * anneal
        lr = opts.lr * (0.5 + 0.5 * (1.0 - step.to_f64 / steps.to_f64))
        bias1 = 1.0 - BETA1 ** (step + 1)
        bias2 = 1.0 - BETA2 ** (step + 1)

        1.upto(nv) do |var|
          g[var] += mu * (1.0 - 2.0 * x[var])
          m[var] = BETA1 * m[var] + (1.0 - BETA1) * g[var]
          v[var] = BETA2 * v[var] + (1.0 - BETA2) * g[var] * g[var]
          mh = m[var] / bias1
          vh = v[var] / bias2
          x[var] -= lr * mh / (Math.sqrt(vh) + 1.0e-8)
          x[var] = 1.0e-4 if x[var] < 1.0e-4
          x[var] = 1.0 - 1.0e-4 if x[var] > 1.0 - 1.0e-4
        end
      end
    end

    class Options
      property time_budget : Float64
      property restarts : Int32
      property steps : Int32
      property repair_flips : Int64
      property seed : UInt64
      property lr : Float64
      property tau : Float64
      property dec_threshold : Float64
      property dec_freq : Int32
      property warm_start : Array(UInt8)?
      property json : Bool

      def initialize(@time_budget = 0.0, @restarts = 4, @steps = 300,
                     @repair_flips = 1_000_000_i64, @seed = 42_u64,
                     @lr = 0.02, @tau = 1.0, @dec_threshold = 0.45,
                     @dec_freq = 50, @warm_start = nil, @json = true)
      end
    end

    struct Result
      getter status : String
      getter hard_unsatisfied : Int64
      getter soft_cost : Float64
      getter assignment : Array(UInt8)
      getter flips : Int64
      getter converged : Bool
      getter restarts : Int32
      getter wall : Float64

      def initialize(@status, @hard_unsatisfied, @soft_cost, @assignment,
                     @flips, @converged, @restarts, @wall)
      end
    end

    module Solver
      extend self

      def solve(f : Formula, opts : Options) : Result
        nv = f.nv
        asn = Array(UInt8).new(nv, 0_u8)
        best = Array(UInt8).new(nv, 0_u8)
        best_h = Int64::MAX
        best_s = Float64::INFINITY
        valid = false
        g = Array(Float64).new(nv + 1, 0.0)
        t0 = Time.monotonic
        restarts_done = 0
        last_flips = 0_i64
        last_conv = false
        stop_all_sat = false

        warm = opts.warm_start
        if warm
          full = Array(UInt8).new(nv + 1, 0_u8)
          1.upto({nv, warm.size - 1}.min) { |i| full[i] = warm[i] }
          warm = full
        end

        r = 0
        loop do
          if opts.time_budget > 0 &&
             (Time.monotonic - t0).total_seconds > opts.time_budget * 0.98
            break
          end
          break if opts.restarts > 0 && r >= opts.restarts
          break if stop_all_sat
          if opts.restarts == 0 && opts.time_budget <= 0 && r >= 1
            break
          end

          seed = opts.seed &* 0x9E3779B97F4A7C15_u64
          seed ^= 0x1000193_u64 &* (r.to_u64 + 1_u64)
          seed ^= 0xDB3FCD2C6B0B71FD_u64
          rng = Rng.new(seed | 1_u64)

          x = Array(Float64).new(nv + 1, 0.0)
          m = Array(Float64).new(nv + 1, 0.0)
          v = Array(Float64).new(nv + 1, 0.0)

          if warm && r == 0
            1.upto(nv) { |i| x[i] = warm.not_nil![i] == 1 ? 0.999 : 0.001 }
          elsif r == 0
            1.upto(nv) { |i| x[i] = 0.5 + 0.4 * (rng.next_double - 0.5) }
          else
            polarity = Array(Float64).new(nv + 1, 0.0)
            f.nc.times do |c|
              (f.offset[c]...f.offset[c + 1]).each do |j|
                lit = f.lits[j]
                var = lit.abs
                polarity[var] += lit > 0 ? f.weight[c] : -f.weight[c]
              end
            end
            mix = 0.5 / (r + 1).to_f64
            1.upto(nv) do |i|
              x[i] = 0.5 + (1.0 - mix) * 0.4 * (rng.next_double - 0.5) +
                     mix * 0.25 * Math.tanh(polarity[i] / 10.0)
            end
          end

          decim = Array(UInt8).new(nv + 1, 0_u8)
          step = 1
          while step <= opts.steps
            if opts.time_budget > 0 &&
               (Time.monotonic - t0).total_seconds > opts.time_budget * 0.9
              break
            end
            Continuous.gradient_step(f, x, g, m, v, step, opts)
            if opts.dec_freq > 0 && step % opts.dec_freq == 0
              1.upto(nv) do |i|
                if decim[i] == 0 && (x[i] - 0.5).abs > opts.dec_threshold
                  x[i] = x[i] > 0.5 ? 0.999 : 0.001
                  decim[i] = 1_u8
                end
              end
            end
            step += 1
          end

          1.upto(nv) { |i| asn[i - 1] = x[i] > 0.5 ? 1_u8 : 0_u8 }

          if warm && r == 0
            h, s = cost(f, asn)
            best = asn.dup
            best_h = h
            best_s = s
            valid = true
          end

          used, conv, repaired = NuWLS.repair(
            f, asn, opts.repair_flips, opts.time_budget, t0,
            opts.seed &+ r.to_u64)
          asn = repaired
          last_flips = used
          last_conv = conv

          h, s = cost(f, asn)
          if !valid || h < best_h || (h == best_h && s < best_s - 1.0e-12)
            best = asn.dup
            best_h = h
            best_s = s
            valid = true
          end
          restarts_done = r + 1
          stop_all_sat = true if h == 0 && s == 0.0

          r += 1
        end

        unless valid
          best_h, best_s = cost(f, best)
        end
        status = if best_h == 0 && best_s == 0.0
                   "SAT"
                 elsif best_h == 0
                   "FEASIBLE"
                 else
                   "UNKNOWN"
                 end
        Result.new(status, best_h, best_s, best,
          last_flips, last_conv || (best_h == 0 && best_s == 0.0),
          restarts_done, (Time.monotonic - t0).total_seconds)
      end

      def cost(f : Formula, asn : Array(UInt8)) : Tuple(Int64, Float64)
        hard = 0_i64
        soft = 0.0
        f.nc.times do |c|
          sat = false
          j = f.offset[c]
          while j < f.offset[c + 1]
            lit = f.lits[j]
            v = lit.abs - 1
            if (lit > 0) == (asn[v] != 0)
              sat = true
              break
            end
            j += 1
          end
          unless sat
            if f.hard[c] == 1
              hard += 1
            else
              soft += f.weight[c]
            end
          end
        end
        {hard, soft}
      end
    end
  end
end
