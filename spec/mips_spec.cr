require "spec"
require "big"
require "../src/weight_index"

private alias MI = Semrember::Mips

private def mips_scores(w : Array(Float32), x : Array(Float32), labels : Array(Int32))
  labels.map do |c|
    value = 0_f32
    x.each_with_index { |v, j| value += w[c*x.size + j] * v }
    value
  end
end

private def mips_reference(w : Array(Float32), x : Array(Float32), t : Int32, k : Int32)
  ids = (0...t).to_a
  scores = mips_scores(w, x, ids)
  ids.sort_by { |i| {-scores[i], i} }.first(k).map { |i| MI::Hit.new(i, scores[i]) }
end

private def mips_check(w, x, t, config, basis = nil)
  index = MI::Index.new(w, t, x.size, config, basis: basis)
  index.bounds(x).each do |ids, bound|
    mips_scores(w, x, ids).each { |value| value.to_f64.should be <= bound }
  end
  (1..Math.min(3, t)).each do |k|
    result = index.search(x, k) { |ids| mips_scores(w, x, ids) }
    unless result.fallback?
      result.hits.should eq(mips_reference(w, x, t, k))
      # Includes signed zero/identical Float32 representations.
      result.hits.each do |hit|
        hit.score.unsafe_as(UInt32).should eq(mips_scores(w, x, [hit.label])[0].unsafe_as(UInt32))
      end
    end
  end
end

describe Semrember::Mips do
  it "widens zero, negative values and neighboring floats correctly" do
    MI::Outward.up(0.0).unsafe_as(UInt64).should eq(1_u64)
    MI::Outward.down(0.0).should be < 0.0
    [-1e300, -1.0, 1e-300, 1.0, 1e300].each do |x|
      MI::Outward.up(x).should be > x
      MI::Outward.down(x).should be < x
    end
  end

  it "checks every node and top-1/2/3 across random sizes, ranks and scales" do
    rng = Random.new(123_u64)
    60.times do |iteration|
      t = rng.rand(4..55)
      d = rng.rand(1..30)
      scale = 10.0 ** rng.rand(-12..12)
      w = Array.new(t*d) { ((rng.rand*2 - 1)*scale).to_f32 }
      x = Array.new(d) { ((rng.rand*2 - 1)/scale).to_f32 }
      config = MI::Config.new(rank: iteration % (d + 1), leaf_size: rng.rand(1..8), power_iters: 1)
      mips_check(w, x, t, config)
    end
  end

  it "handles nonorthonormal and rank-deficient supplied bases" do
    w = [1, 2, 3, 1, 2, 3, -4, -3, -2, 0, 0, 0].map(&.to_f32)
    x = [0.25_f32, -0.5_f32, 2_f32]
    [[[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]], [[1e6, 0.0, 0.0]], [] of Array(Float64)].each do |basis|
      mips_check(w, x, 4, MI::Config.new(leaf_size: 1), basis)
    end
  end

  it "preserves duplicate rows, all-negative scores and lower-label ties" do
    w = [-1, -2, -1, -2, -3, -4, -1, -2, 0, -8].map(&.to_f32)
    mips_check(w, [1_f32, 1_f32], 5, MI::Config.new(rank: 2, leaf_size: 1))
    index = MI::Index.new(w, 5, 2, MI::Config.new(rank: 0, leaf_size: 1))
    result = index.search([1_f32, 1_f32], 3) { |ids| mips_scores(w, [1_f32, 1_f32], ids) }
    result.hits.map(&.label).should eq([0, 1, 3])
  end

  it "bounds subnormal products and cancellation using an exact rational dot" do
    tiny = 1_u32.unsafe_as(Float32)
    w = [tiny, -tiny, 1_f32, 1e20_f32, -1e20_f32, tiny,
         tiny, tiny, -1_f32, -1e20_f32, 1e20_f32, -tiny]
    x = [0.5_f32, 0.5_f32, tiny, 1_f32, 1_f32, 1_f32]
    index = MI::Index.new(w, 2, 6, MI::Config.new(rank: 0, leaf_size: 1))
    index.bounds(x).each do |ids, upper|
      ids.each do |id|
        exact = BigRational.new(0)
        absolute = BigRational.new(0)
        x.each_with_index do |v, j|
          term = BigRational.new(w[id*6 + j].to_f64) * BigRational.new(v.to_f64)
          exact += term
          absolute += term.abs
        end
        observed = mips_scores(w, x, [id])[0]
        observed.to_f64.should be <= upper
        ku = BigRational.new(12, 1 << 24)
        error = ku / (1 - ku)*absolute + BigRational.new(12, BigInt.new(2)**149)/(1 - ku)
        (BigRational.new(observed.to_f64) - exact).abs.should be <= error
      end
    end
  end

  it "requests a full scan for unsafe arithmetic, zero queries and stale versions" do
    w = [1_f32, 2_f32, 3_f32, 4_f32]
    index = MI::Index.new(w, 4, 1, version: 7_i64)
    [[0_f32], [Float32::NAN], [Float32::INFINITY], [Float32::MAX]].each do |x|
      index.search(x, 1) { |ids| mips_scores(w, x, ids) }.fallback?.should be_true
    end
    called = false
    result = index.search([1_f32], 1, expected_version: 8_i64) { |ids| called = true; mips_scores(w, [1_f32], ids) }
    result.fallback?.should be_true
    called.should be_false
    expect_raises(ArgumentError) { MI::Index.new([Float32::NAN], 1, 1) }
  end

  it "discards partial answers on node or label budget exhaustion" do
    w = (1..20).map(&.to_f32)
    [MI::Config.new(max_nodes: 0), MI::Config.new(leaf_size: 2, max_labels: 1)].each do |config|
      index = MI::Index.new(w, 20, 1, config)
      result = index.search([1_f32], 3) { |ids| mips_scores(w, [1_f32], ids) }
      result.fallback?.should be_true
      result.hits.should be_empty
    end
  end

  it "handles empty shapes, top-k limits and invalid arguments explicitly" do
    empty = MI::Index.new([] of Float32, 0, 0)
    empty.search([] of Float32, 1) { |ids| [] of Float32 }.hits.should be_empty
    single = MI::Index.new([1_f32], 1, 1)
    single.search([1_f32], 0) { |ids| [1_f32] }.hits.should be_empty
    single.search([1_f32], 1) { |ids| [1_f32] }.fallback?.should be_true
    expect_raises(ArgumentError) { single.search([1_f32], -1) { |ids| [1_f32] } }
    expect_raises(ArgumentError) { MI::Index.new([1_f32], 2, 2) }
    MI::Index.new([] of Float32, 3, 0).search([] of Float32, 1) { |ids| [] of Float32 }.fallback?.should be_true
  end

  it "owns its geometry independently of mutable construction arrays" do
    w = (1..32).map(&.to_f32)
    original = w.dup
    basis = [[1.0]]
    index = MI::Index.new(w, 32, 1, MI::Config.new(leaf_size: 2), basis: basis)
    w.fill(-99_f32)
    basis[0][0] = 1e50
    result = index.search([1_f32], 3) { |ids| mips_scores(original, [1_f32], ids) }
    result.hits.should eq(mips_reference(original, [1_f32], 32, 3))
    result.labels_scored.should be < 32
  end

  it "actually prunes separated low-rank rows with the generated basis" do
    t, d = 128, 16
    w = Array.new(t*d) { |ij| ((ij // d + 1) * (ij % d + 1)).to_f32 }
    x = Array.new(d, 1_f32)
    index = MI::Index.new(w, t, d, MI::Config.new(rank: 4, leaf_size: 4))
    result = index.search(x, 3) { |ids| mips_scores(w, x, ids) }
    result.fallback?.should be_false
    result.hits.should eq(mips_reference(w, x, t, 3))
    result.labels_scored.should be < t // 2
  end

  it "encloses exact rational dots even when basis reconstruction cancels" do
    w = [1_f32, -1_f32, 1e-10_f32, -2_f32, 2_f32, -1e-10_f32,
         3_f32, -3_f32, 0_f32, 0_f32, 0_f32, 0_f32]
    x = [1_f32, 1_f32, 0.25_f32]
    basis = [[1e8, 1e8, 1e-8], [1e8, 1e8 + 1, -1e-8]]
    index = MI::Index.new(w, 4, 3, MI::Config.new(leaf_size: 1), basis: basis)
    index.bounds(x).each do |ids, upper|
      ids.each do |id|
        exact = BigRational.new(0)
        3.times { |j| exact += BigRational.new(w[id*3 + j].to_f64) * BigRational.new(x[j].to_f64) }
        exact.should be <= BigRational.new(upper)
      end
    end
  end

  it "falls back when the error denominator is invalid or the callback is nonfinite" do
    w = (1..8).map(&.to_f32)
    index = MI::Index.new(w, 8, 1)
    index.search([1_f32], 1, terms: Int32::MAX) { |ids| mips_scores(w, [1_f32], ids) }.fallback?.should be_true
    index.search([1_f32], 1) { |ids| Array.new(ids.size, Float32::NAN) }.fallback?.should be_true
    expect_raises(ArgumentError) { index.search([1_f32], 1) { |ids| [] of Float32 } }
  end

  it "orders equal heap bounds deterministically" do
    heap = MI::Heap.new
    heap.push(2.0, 8)
    heap.push(3.0, 9)
    heap.push(2.0, 2)
    heap.pop.should eq({3.0, 9})
    heap.pop.should eq({2.0, 2})
    heap.pop.should eq({2.0, 8})
    heap.empty?.should be_true
  end
end

describe Semrember::Linalg do
  it "computes a symmetric eigensystem and drops dependent columns" do
    a = [3.0, 1.0, 1.0, 3.0]
    values, vectors = Semrember::Linalg.jacobi_eigen(a, 2)
    values[0].should be_close(4.0, 1e-10)
    values[1].should be_close(2.0, 1e-10)
    2.times do |k|
      2.times do |i|
        av = (0...2).sum { |j| a[i*2 + j] * vectors[j*2 + k] }
        av.should be_close(values[k] * vectors[i*2 + k], 1e-10)
      end
    end
    q = Semrember::Linalg.orthonormalize([[1.0, 2.0], [2.0, 4.0]])
    q.size.should eq(1)
    q[0].sum { |x| x*x }.should be_close(1.0, 1e-12)
    Semrember::Linalg.transpose([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).should eq([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
  end
end
