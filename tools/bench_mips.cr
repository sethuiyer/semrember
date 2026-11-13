require "option_parser"
require "json"
require "../src/weight_index"

alias Mips = Semrember::Mips

t, d, rank, leaf, queries, topk, seed = 1024, 4096, 32, 16, 50, 3, 0
repeats = 5
mode = "low-rank"
OptionParser.parse do |p|
  p.banner = "Standalone label-scoring benchmark (not end-to-end server latency)"
  p.on("--labels N", "Labels") { |v| t = v.to_i }
  p.on("--dim N", "Features") { |v| d = v.to_i }
  p.on("--rank N", "Projection rank") { |v| rank = v.to_i }
  p.on("--leaf N", "Leaf size") { |v| leaf = v.to_i }
  p.on("--repeats N", "Timing repetitions (alternating order)") { |v| repeats = v.to_i }
  p.on("--queries N", "Queries") { |v| queries = v.to_i }
  p.on("--topk N", "Top k") { |v| topk = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_i }
  p.on("--mode NAME", "low-rank|gaussian|duplicates|sparse") { |v| mode = v }
  p.on("--help", "Help") { puts p; exit }
end
raise "invalid dimensions/options" unless t > 0 && d > 0 && queries > 0 && topk > 0 && topk <= t && seed >= 0 && repeats > 0
raise "unknown mode" unless {"low-rank", "gaussian", "duplicates", "sparse"}.includes?(mode)
rng = Random.new(seed.to_u64)
normal = -> { Math.sqrt(-2.0 * Math.log(Math.max(rng.rand, 1e-100))) * Math.cos(2.0 * Math::PI * rng.rand) }
axes = Array.new(4) { Array.new(d) { normal.call.to_f32 } }
w = Array.new(t*d) do |ij|
  i, j = ij // d, ij % d
  case mode
  when "low-rank"
    (axes[i % 4][j] * (0.5 + (i // 4).to_f64 / t)).to_f32
  when "duplicates"
    (axes[i % 4][j] + normal.call * 1e-5).to_f32
  when "sparse"
    rng.rand < 0.98 ? 0_f32 : normal.call.to_f32
  else
    normal.call.to_f32
  end
end
xs = Array.new(queries) do |q|
  Array.new(d) do |j|
    if mode == "low-rank" || mode == "duplicates"
      (axes[q % 4][j] + normal.call*0.05).to_f32
    else
      normal.call.to_f32
    end
  end
end
score = ->(x : Array(Float32), ids : Array(Int32)) {
  # Same feature-major update order as Memory#scores.
  out = Array.new(ids.size, 0_f32)
  x.each_with_index do |v, j|
    ids.each_with_index { |id, k| out[k] += w[id*d + j] * v }
  end
  out
}
all = (0...t).to_a
full = ->(x : Array(Float32)) {
  # Full scan mirrors Memory#scores directly, without an ids indirection.
  values = Array.new(t, 0_f32)
  x.each_with_index do |v, j|
    t.times { |c| values[c] += w[c*d + j] * v }
  end
  all.sort_by { |i| {-values[i], i} }.first(topk).map { |i| Mips::Hit.new(i, values[i]) }
}
config = Mips::Config.new(rank: rank, leaf_size: leaf, seed: seed, max_labels: t, max_nodes: t*2)
start = Time.monotonic
index = Mips::Index.new(w, t, d, config)
build_s = (Time.monotonic - start).total_seconds
# Warm both paths before timing, then validate outside timed regions.
xs.first(3).each do |x|
  full.call(x)
  index.search(x, topk) { |ids| score.call(x, ids) }
end
references = [] of Array(Mips::Hit)
actual = [] of Array(Mips::Hit)
results = [] of Mips::SearchResult
full_times = [] of Float64
indexed_times = [] of Float64
repeats.times do |rep|
  order = rep.even? ? [false, true] : [true, false]
  order.each do |indexed|
    start = Time.monotonic
    if indexed
      results = xs.map { |x| index.search(x, topk) { |ids| score.call(x, ids) } }
      # Include mandatory complete scans after fallback in the timed path.
      actual = results.map_with_index { |r, i| r.fallback? ? full.call(xs[i]) : r.hits }
      indexed_times << (Time.monotonic - start).total_seconds
    else
      references = xs.map { |x| full.call(x) }
      full_times << (Time.monotonic - start).total_seconds
    end
  end
  raise "top-k mismatch" unless actual == references
end
median = ->(values : Array(Float64)) {
  sorted = values.sort
  (sorted[(sorted.size - 1)//2] + sorted[sorted.size//2]) / 2
}
full_s = median.call(full_times)
indexed_s = median.call(indexed_times)
puts({mode: mode, labels: t, dimension: d, requested_rank: rank, actual_rank: index.rank,
      leaf: leaf, queries: queries, repeats: repeats, full_samples: full_times, indexed_samples: indexed_times, topk: topk, seed: seed, build_seconds: build_s,
      numeric_payload_bytes: index.payload_bytes, full_seconds: full_s, indexed_seconds: indexed_s,
      scoring_speedup: full_s / indexed_s, mean_nodes: results.sum(&.nodes_visited).to_f64 / queries,
      mean_labels: results.sum(&.labels_scored).to_f64 / queries,
      fallback_rate: results.count(&.fallback?).to_f64 / queries, parity: true,
      scope: "standalone scoring; no end-to-end speedup claimed"}.to_json)
