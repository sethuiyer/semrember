require "spec"
require "../src/maxsat"

private def parse(text : String) : Semrember::MaxSat::Formula
  Semrember::MaxSat::Wcnf.parse(text)
end

private def brute(f : Semrember::MaxSat::Formula) : Tuple(Int64, Float64)
  best_h = Int64::MAX
  best_s = Float64::INFINITY
  (0...(1 << f.nv)).each do |mask|
    asn = Array(UInt8).new(f.nv, 0_u8)
    f.nv.times { |v| asn[v] = ((mask >> v) & 1).to_u8 }
    h, s = Semrember::MaxSat::Solver.cost(f, asn)
    if h < best_h || (h == best_h && s < best_s)
      best_h = h
      best_s = s
    end
  end
  {best_h, best_s}
end

private def random_wcnf(rng : Random, nv : Int32, nc : Int32,
                        hard_fraction : Float64) : String
  String.build do |io|
    io << "p wcnf " << nv << " " << nc << " 100000\n"
    nc.times do |i|
      width = 2 + rng.rand(2)
      vars = (0...nv).to_a.sample(width, rng)
      lits = vars.map { |v| rng.rand(2) == 0 ? v + 1 : -(v + 1) }
      if rng.rand < hard_fraction
        io << "h " << lits.join(" ") << " 0\n"
      else
        io << (1 + rng.rand(9)) << " " << lits.join(" ") << " 0\n"
      end
    end
  end
end

describe Semrember::MaxSat do
  it "parses CNF, WCNF, h-lines, top weights and zero-weight hard clauses" do
    f = parse("p wcnf 3 3 10\n10 1 2 0\nh -1 3 0\n2 -2 -3 0\n")
    f.nv.should eq 3
    f.nc.should eq 3
    f.hard.should eq [1_u8, 1_u8, 0_u8]
    f.weight.should eq [10.0, 1.0, 2.0]

    f2 = parse("p wcnf 2 1 100\n0 1 -2 0\n")
    f2.hard[0].should eq 1_u8
  end

  it "keeps incremental scores equal to a full rebuild" do
    f = parse("p wcnf 4 8 100\n" \
              "h 3 4 0\nh -3 1 0\nh -4 2 0\nh -3 -4 0\n" \
              "1 -3 0\n1 -4 0\n5 -1 0\n10 -2 0\n")
    asn = Array(UInt8).new(f.nv, 0_u8)
    state = Semrember::MaxSat::SearchState.new(f, asn, 7_u64)
    max = 0.0
    500.times do |i|
      v = state.pick
      break if v < 0
      state.flip(asn, v, i.to_i64 + 1)
      snapshot = state.score.dup
      state.rebuild_scores
      f.nv.times do |j|
        diff = (snapshot[j] - state.score[j]).abs
        max = diff if diff > max
      end
    end
    max.should be < 1e-6
  end

  it "keeps scores consistent after additive weight updates" do
    f = parse("p wcnf 4 8 100\n" \
              "h 3 4 0\nh -3 1 0\nh -4 2 0\nh -3 -4 0\n" \
              "1 -3 0\n1 -4 0\n5 -1 0\n10 -2 0\n")
    asn = Array(UInt8).new(f.nv, 0_u8)
    state = Semrember::MaxSat::SearchState.new(f, asn, 9_u64)
    max = 0.0
    40.times do
      state.update_weights
      snapshot = state.score.dup
      state.rebuild_scores
      f.nv.times do |j|
        diff = (snapshot[j] - state.score[j]).abs
        max = diff if diff > max
      end
    end
    max.should be < 1e-6
  end

  it "returns a witness whose recounted cost matches the reported cost" do
    rng = Random.new(13)
    10.times do
      f = parse(random_wcnf(rng, 8, 18, 0.4))
      result = Semrember::MaxSat::Solver.solve(f,
        Semrember::MaxSat::Options.new(time_budget: 1.0, restarts: 3,
          steps: 80, repair_flips: 20_000_i64, seed: 4_u64))
      h, s = Semrember::MaxSat::Solver.cost(f, result.assignment)
      h.should eq result.hard_unsatisfied
      s.should be_close(result.soft_cost, 1e-9)
    end
  end

  it "never beats the brute-force optimum and finds it on small instances" do
    rng = Random.new(17)
    8.times do
      f = parse(random_wcnf(rng, 5, 12, 0.3))
      optimum_h, optimum_s = brute(f)
      result = Semrember::MaxSat::Solver.solve(f,
        Semrember::MaxSat::Options.new(time_budget: 1.0, restarts: 4,
          steps: 60, repair_flips: 30_000_i64, seed: 11_u64))
      result.hard_unsatisfied.should be >= optimum_h
      if result.hard_unsatisfied == optimum_h
        result.soft_cost.should be >= optimum_s - 1e-9
      end
      result.hard_unsatisfied.should eq optimum_h
      if optimum_h == 0
        result.soft_cost.should eq optimum_s
      end
    end
  end

  it "is lexicographic: a hard clause beats any soft cost" do
    f = parse("p wcnf 1 2 100\nh 1 0\n50 -1 0\n")
    result = Semrember::MaxSat::Solver.solve(f,
      Semrember::MaxSat::Options.new(time_budget: 1.0, seed: 2_u64))
    result.hard_unsatisfied.should eq 0
    result.soft_cost.should eq 50.0
  end

  it "is deterministic for a fixed seed" do
    f = parse("p wcnf 4 8 100\n" \
              "h 3 4 0\nh -3 1 0\nh -4 2 0\nh -3 -4 0\n" \
              "1 -3 0\n1 -4 0\n5 -1 0\n10 -2 0\n")
    opts = Semrember::MaxSat::Options.new(time_budget: 1.0, restarts: 2,
      steps: 50, repair_flips: 10_000_i64, seed: 5_u64)
    a = Semrember::MaxSat::Solver.solve(f, opts)
    b = Semrember::MaxSat::Solver.solve(f, opts)
    a.hard_unsatisfied.should eq b.hard_unsatisfied
    a.soft_cost.should eq b.soft_cost
    a.assignment.should eq b.assignment
  end

  it "handles an empty formula" do
    f = parse("p cnf 2 0\n")
    result = Semrember::MaxSat::Solver.solve(f,
      Semrember::MaxSat::Options.new(time_budget: 0.2, seed: 1_u64))
    result.status.should eq "SAT"
    result.hard_unsatisfied.should eq 0
    result.soft_cost.should eq 0.0
    result.assignment.size.should eq 2
  end

  it "registers a warm start as the incumbent even with no continuous steps" do
    f = parse("p wcnf 4 8 100\n" \
              "h 3 4 0\nh -3 1 0\nh -4 2 0\nh -3 -4 0\n" \
              "1 -3 0\n1 -4 0\n5 -1 0\n10 -2 0\n")
    # 1-based DIMACS witness: a1=true, a2=false, r1=true, r2=false
    warm = [0_u8, 1_u8, 0_u8, 1_u8, 0_u8]
    result = Semrember::MaxSat::Solver.solve(f,
      Semrember::MaxSat::Options.new(time_budget: 1.0, restarts: 1,
        steps: 0, repair_flips: 0_i64, seed: 3_u64, warm_start: warm))
    result.hard_unsatisfied.should eq 0
    result.soft_cost.should eq 6.0
  end

  it "keeps the continuous relaxation finite and in range" do
    f = parse("p wcnf 3 4 10\nh 1 2 0\n2 -1 3 0\n1 -2 -3 0\n3 1 -3 0\n")
    nv = f.nv
    x = Array(Float64).new(nv + 1, 0.5)
    g = Array(Float64).new(nv + 1, 0.0)
    m = Array(Float64).new(nv + 1, 0.0)
    v = Array(Float64).new(nv + 1, 0.0)
    opts = Semrember::MaxSat::Options.new(steps: 50)
    1.upto(50) do |step|
      Semrember::MaxSat::Continuous.gradient_step(f, x, g, m, v, step, opts)
    end
    1.upto(nv) do |i|
      x[i].finite?.should be_true
      x[i].should be >= 0.0
      x[i].should be <= 1.0
    end
  end
end
