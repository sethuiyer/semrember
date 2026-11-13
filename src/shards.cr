# shards.cr — a small, exact-evidence shard planner.
#
# This is the first SUTRA architecture slice:
#   * build label shards with the parent's frozen feature map
#   * measure which shards preserve each query's top-1 answer
#   * encode deployment + per-query assignment as weighted MaxSAT
#   * solve with the Crystal NuWLS core
#   * train a compact Semrember router on the selected assignments
#
# It is deliberately conservative: top-1 fidelity only, a full-model
# fallback that is assumed resident, and a budget measured in retained
# observations. Escalation/top-2 fidelity and observation clustering are
# later slices.
require "./semrember"
require "./maxsat"

module Semrember
  module Shards
    extend self

    record Candidate, id : Int32, label : String, model : Memory, memory : Int32

    record Encoding, formula : MaxSat::Formula, deploy : Array(Int32),
      rs : Array(Array(Int32)), units : Array(Array(Int32)),
      atmost : Array(Array(Int32))

    struct Plan
      getter selected : Array(String)
      getter assignments : Array(Tuple(String, String))
      getter fallback_count : Int32
      getter router_accuracy : Float64
      getter solver_status : String
      getter solver_cost : Float64
      getter valid : Bool
      getter budget_used : Int32
      getter budget : Int32
      getter candidate_count : Int32

      def initialize(@selected, @assignments, @fallback_count, @router_accuracy,
                     @solver_status, @solver_cost, @valid, @budget_used,
                     @budget, @candidate_count)
      end
    end

    private class Builder
      getter nv : Int32
      getter offset : Array(Int32)
      getter lits : Array(Int32)
      getter weights : Array(Float64)
      getter hard : Array(UInt8)

      def initialize
        @nv = 0
        @offset = [0_i32]
        @lits = [] of Int32
        @weights = [] of Float64
        @hard = [] of UInt8
      end

      def var : Int32
        @nv += 1
        @nv
      end

      def clause(clause_lits : Array(Int32), is_hard : Bool, weight : Float64 = 1.0) : Nil
        clause_lits.each { |lit| @lits << lit }
        @offset << @lits.size
        @weights << weight
        @hard << (is_hard ? 1_u8 : 0_u8)
      end

      def formula : MaxSat::Formula
        MaxSat::Formula.new(@nv, @offset, @lits, @weights, @hard)
      end
    end

    # Sinz-style sequential at-most-k over positive unit variables.
    private def add_at_most_k(b : Builder, xs : Array(Int32), k : Int32) : Array(Array(Int32))
      if k < 0
        b.clause([] of Int32, true)
        return [] of Array(Int32)
      end
      return [] of Array(Int32) if k >= xs.size
      if k == 0
        xs.each { |x| b.clause([-x], true) }
        return [] of Array(Int32)
      end
      n = xs.size
      s = Array(Array(Int32)).new(n - 1) { Array(Int32).new(k, 0) }
      (n - 1).times { |i| k.times { |j| s[i][j] = b.var } }
      # A one-element prefix cannot have count >= 2.
      (1...k).each { |j| b.clause([-s[0][j]], true) }
      # x_i -> prefix count >= 1
      (0...(n - 1)).each { |i| b.clause([-xs[i], s[i][0]], true) }
      # prefix counts persist and advance
      (1...(n - 1)).each do |i|
        k.times { |j| b.clause([-s[i - 1][j], s[i][j]], true) }
        (1...k).each do |j|
          b.clause([-xs[i], -s[i - 1][j - 1], s[i][j]], true)
        end
      end
      # x_i plus a prefix already at k is forbidden
      (1...n).each { |i| b.clause([-xs[i], -s[i - 1][k - 1]], true) }
      s
    end

    private def top_label(mem : Memory, query : String) : String?
      hit = mem.query(query, 1)[0]?
      hit ? hit[0] : nil
    end

    def label_candidates(parent : Memory) : Array(Candidate)
      candidates = [] of Candidate
      parent.labels.each do |label|
        child = parent.subset_by_labels([label])
        next if child.stored == 0
        candidates << Candidate.new(candidates.size, label, child, child.stored)
      end
      candidates
    end

    private def encode(candidates : Array(Candidate), h : Array(Array(UInt8)),
                       budget : Int32, fallback_cost : Float64,
                       label_cost : Float64) : Encoding
      b = Builder.new
      qcount = h.size
      deploy = Array(Int32).new(candidates.size) { b.var }
      rs = Array(Array(Int32)).new(qcount) do |q|
        Array(Int32).new(candidates.size + 1) { b.var }
      end

      qcount.times do |q|
        # Exactly one destination: fallback is destination 0.
        b.clause(rs[q], true)
        (0...(rs[q].size - 1)).each do |i|
          ((i + 1)...rs[q].size).each do |j|
            b.clause([-rs[q][i], -rs[q][j]], true)
          end
        end
        b.clause([-rs[q][0]], false, fallback_cost)
        candidates.size.times do |s|
          if h[q][s] == 0
            b.clause([-rs[q][s + 1]], true)
          end
          b.clause([-rs[q][s + 1], deploy[s]], true)
          b.clause([-rs[q][s + 1]], false, label_cost)
        end
      end

      candidates.size.times do |s|
        memory = candidates[s].memory
        if memory > 0
          b.clause([-deploy[s]], false, memory.to_f64)
        end
      end

      # Sum of retained observations over deployed shards <= budget.
      units = [] of Int32
      units_by_s = Array(Array(Int32)).new(candidates.size) { [] of Int32 }
      candidates.each_with_index do |candidate, s|
        candidate.memory.times do
          u = b.var
          units << u
          units_by_s[s] << u
          b.clause([-deploy[s], u], true)
          b.clause([-u, deploy[s]], true)
        end
      end
      atmost = add_at_most_k(b, units, budget)

      Encoding.new(b.formula, deploy, rs, units_by_s, atmost)
    end

    private def greedy_select(candidates : Array(Candidate), h : Array(Array(UInt8)),
                              budget : Int32, fallback_cost : Float64,
                              label_cost : Float64) : Tuple(Array(Int32), Array(Int32))
      qcount = h.size
      scount = candidates.size
      cover = Array(Int32).new(scount, 0)
      qcount.times do |q|
        scount.times { |s| cover[s] += 1 if h[q][s] == 1 }
      end
      order = (0...scount).to_a.sort_by do |s|
        -((fallback_cost - label_cost) * cover[s] - candidates[s].memory)
      end
      selected = [] of Int32
      used = 0
      order.each do |s|
        next if cover[s] == 0
        next if (fallback_cost - label_cost) * cover[s] - candidates[s].memory <= 0
        next if used + candidates[s].memory > budget
        selected << s
        used += candidates[s].memory
      end
      assigned = Array(Int32).new(qcount, 0)
      qcount.times do |q|
        selected.each do |s|
          if h[q][s] == 1
            assigned[q] = s + 1
            break
          end
        end
      end
      {selected, assigned}
    end

    private def warm_assignment(formula : MaxSat::Formula,
                                candidates : Array(Candidate), deploy : Array(Int32),
                                rs : Array(Array(Int32)), units_by_s : Array(Array(Int32)),
                                atmost : Array(Array(Int32)), selected : Array(Int32),
                                assigned : Array(Int32), budget : Int32) : Array(UInt8)
      warm = Array(UInt8).new(formula.nv + 1, 0_u8)
      selected_set = selected.to_set
      candidates.size.times do |s|
        on = selected_set.includes?(s) ? 1_u8 : 0_u8
        warm[deploy[s]] = on
        units_by_s[s].each { |u| warm[u] = on }
      end
      rs.size.times do |q|
        chosen = assigned[q]
        (0..candidates.size).each do |d|
          warm[rs[q][d]] = d == chosen ? 1_u8 : 0_u8
        end
      end
      flat = units_by_s.flatten
      if budget > 0 && budget < flat.size && !atmost.empty?
        atmost.each_with_index do |row, i|
          count = (0..i).count { |j| warm[flat[j]] == 1 }
          row.each_with_index { |v, j| warm[v] = count >= j + 1 ? 1_u8 : 0_u8 }
        end
      end
      warm
    end

    def build_plan(parent : Memory, queries : Array(String), budget : Int32,
                   time_budget : Float64 = 2.0, seed : UInt64 = 1_u64,
                   fallback_cost : Float64 = 10.0,
                   label_cost : Float64 = 1.0) : Plan
      raise ArgumentError.new("queries must not be empty") if queries.empty?
      candidates = label_candidates(parent)
      h = Array(Array(UInt8)).new(queries.size) do |q|
        full = top_label(parent, queries[q])
        Array(UInt8).new(candidates.size) do |s|
          (full && top_label(candidates[s].model, queries[q]) == full) ? 1_u8 : 0_u8
        end
      end

      encoded = encode(candidates, h, budget, fallback_cost, label_cost)
      formula = encoded.formula
      deploy = encoded.deploy
      rs = encoded.rs
      selected_hint, assigned_hint = greedy_select(candidates, h, budget, fallback_cost, label_cost)
      warm = warm_assignment(formula, candidates, deploy, rs, encoded.units,
        encoded.atmost, selected_hint, assigned_hint, budget)
      options = MaxSat::Options.new(time_budget: time_budget, restarts: 4,
        steps: 100, repair_flips: 500_000_i64, seed: seed, warm_start: warm)
      result = MaxSat::Solver.solve(formula, options)
      asn = result.assignment
      value = ->(id : Int32) { id >= 1 && id <= asn.size && asn[id - 1] == 1 }

      selected = [] of Int32
      candidates.size.times { |s| selected << s if value.call(deploy[s]) }
      budget_used = selected.sum { |s| candidates[s].memory }

      assignments = [] of Tuple(String, String)
      fallback_count = 0
      valid = result.hard_unsatisfied == 0 && budget_used <= budget
      queries.each_with_index do |query, q|
        chosen = [] of Int32
        (0..candidates.size).each { |d| chosen << d if value.call(rs[q][d]) }
        if chosen.size != 1
          valid = false
          assignments << {query, "fallback"}
          fallback_count += 1
          next
        end
        d = chosen[0]
        if d == 0
          fallback_count += 1
          assignments << {query, "fallback"}
        else
          s = d - 1
          valid = false if h[q][s] == 0 || !value.call(deploy[s])
          assignments << {query, "label:#{candidates[s].label}"}
        end
      end

      # Train a compact router on the assignment targets, then evaluate the
      # router against the full model on the same queries.
      targets = {} of String => String
      assignments.each { |(query, dest)| targets[query] = dest }
      router = Memory.new.fit(targets)
      by_dest = {} of String => Candidate
      candidates.each { |candidate| by_dest["label:#{candidate.label}"] = candidate }
      router_ok = 0
      queries.each_with_index do |query, q|
        predicted = router.query(query, 1)[0]?
        dest = predicted ? predicted[0] : "fallback"
        model = if dest == "fallback"
                  parent
                elsif candidate = by_dest[dest]?
                  candidate.model
                else
                  parent
                end
        out = top_label(model, query)
        router_ok += 1 if out == top_label(parent, query)
      end
      router_accuracy = queries.size > 0 ? router_ok.to_f64 / queries.size : 0.0

      Plan.new(
        selected.map { |s| candidates[s].label },
        assignments,
        fallback_count,
        router_accuracy,
        result.status,
        result.soft_cost,
        valid,
        budget_used,
        budget,
        candidates.size,
      )
    end
  end
end

