require "../src/semrember"
require "../src/csv"
require "../src/influence"

# Hot-working-set retention experiment.
#
#   crystal run --release tools/retention_experiment.cr -- ~/semrember/build/pokemon.csv
#
# Train on an 80% split, build a perturbed query corpus from the training keys,
# rank memories by several strategies, and measure decision agreement with the
# complete model on the held-out 20%.

path = ARGV[0]? || File.expand_path("~/semrember/build/pokemon.csv")
data = {} of String => String
Semrember::Csv.load_pairs(path, "Name", "Type 1").each do |(name, type)|
  data[name.downcase] = type
end
raise "empty corpus" if data.empty?

items = data.to_a
items.shuffle(Random.new(0))
test_size = {1, items.size // 5}.max
test = items.first(test_size)
train = items[test_size..].to_h
train_keys = train.keys
n = train_keys.size
raise "corpus too small" if n < 20

full = Semrember::Memory.new(solver: :dual, min_df: 1).fit(train)
full_pred = test.map { |(q, _)| full.query(q, 1)[0]? ? full.query(q, 1)[0][0] : "" }
full_escalate = test.map { |(q, _)| full.novelty(q) > 0.55 }

rng = Random.new(19)
query_corpus = Array(String).new(100) do
  k = train_keys[rng.rand(n)].split(' ')
  if k.size > 1 && rng.rand(2) == 0
    k[rng.rand(k.size)] = "mut#{rng.rand(10_000)}"
  else
    k.join
  end
  k.join(' ')
end

profile = Semrember::BanzhafProfile.new
influence = full.banzhaf_influence(query_corpus, 0, profile)
influence_by_key = influence.to_h { |r| {r.key, r.mean_influence} }
leverage = full.leverage_scores
leverage_by_key = train_keys.each_with_index.to_h { |k, i| {k, leverage[i]} }

frequency = Hash(String, Int32).new(0)
query_corpus.each do |q|
  hit = full.similar(q, 1)[0]?
  frequency[hit[0]] += 1 if hit
end

recency = train_keys.each_with_index.to_h { |k, i| {k, i.to_f64 / (n - 1)} }
banzhaf_max = influence_by_key.values.max? || 1.0
banzhaf_norm = influence_by_key.transform_values { |v| banzhaf_max > 0 ? v / banzhaf_max : 0.0 }
hybrid = train_keys.to_h do |k|
  {k, 0.7 * (banzhaf_norm[k]? || 0.0) + 0.3 * (recency[k]? || 0.0)}
end
random_score = train_keys.to_h { |k| {k, rng.rand} }

strategies = {
  "recency"   => recency,
  "random"    => random_score,
  "frequency" => train_keys.to_h { |k| {k, frequency[k].to_f64} },
  "leverage"  => leverage_by_key,
  "banzhaf"   => influence_by_key,
  "hybrid"    => hybrid,
}

puts "pokemon retention: train=#{n} test=#{test.size} queries=#{query_corpus.size}"
puts "influence: total=#{profile.total_ms.round(1)}ms matmul=#{profile.matmul_ms.round(1)}ms"
puts
puts "%-8s %-10s %-12s %-12s" % ["K", "strategy", "decision", "escalation"]
ks = [10, 20, 40, 80, 120, 200, 300, 400, 500, n].select { |k| k <= n }.uniq
ks.each do |k|
  strategies.each do |name, scores|
    keep = train_keys.sort_by { |key| -(scores[key]? || 0.0) }.first(k)
    subset = Semrember::Memory.new(solver: :dual, min_df: 1).fit(train.select { |key, _| keep.includes?(key) })
    decision = 0
    escalation = 0
    test.each_with_index do |(q, _), i|
      hit = subset.query(q, 1)[0]?
      decision += 1 if hit && hit[0] == full_pred[i]
      escalation += 1 if (subset.novelty(q) > 0.55) == full_escalate[i]
    end
    puts "%-8d %-10s %-12.3f %-12.3f" % [k, name, decision.to_f64 / test.size, escalation.to_f64 / test.size]
  end
  puts
end
