require "../src/semrember"
require "../src/csv"
require "../src/influence"

# Real-corpus hot-working-set retention experiment.
#
#   crystal run --release tools/retention_experiment.cr -- \
#     ~/Documents/Documents/Workspace/Datasets/BBC_News_Article_Train.csv \
#     --text-col text --label-col category --max-rows 1200 --queries 100
#
# Or the default Pokemon CSV when no path is supplied.

text_col = "text"
label_col = "category"
max_rows = 0
query_count = 100
seed = 19_u64
path : String? = nil
args = ARGV.dup
i = 0
while i < args.size
  case args[i]
  when "--text-col"
    text_col = args[i + 1]; i += 2
  when "--label-col"
    label_col = args[i + 1]; i += 2
  when "--max-rows"
    max_rows = args[i + 1].to_i; i += 2
  when "--queries"
    query_count = args[i + 1].to_i; i += 2
  when "--seed"
    seed = args[i + 1].to_u64; i += 2
  else
    path = args[i]; i += 1
  end
end
path ||= File.expand_path("~/semrember/build/pokemon.csv")
if path.ends_with?("pokemon.csv")
  text_col = "Name"
  label_col = "Type 1"
end

data = {} of String => String
Semrember::Csv.load_pairs(path, text_col, label_col).each do |(text, label)|
  data[text.downcase] = label
end
raise "empty corpus" if data.empty?

if max_rows > 0 && data.size > max_rows
  data = data.to_a.shuffle(Random.new(seed)).first(max_rows).to_h
end

items = data.to_a
items.shuffle(Random.new(seed))
test_size = {1, items.size // 5}.max
test = items.first(test_size)
train = items[test_size..].to_h
train_keys = train.keys
n = train_keys.size
raise "corpus too small" if n < 20

full = Semrember::Memory.new(solver: :dual, min_df: 1).fit(train)
full_pred = test.map { |(q, _)| full.query(q, 1)[0]? ? full.query(q, 1)[0][0] : "" }
full_escalate = test.map { |(q, _)| full.novelty(q) > 0.55 }

rng = Random.new(seed)
query_corpus = Array(String).new(query_count) do
  text = train_keys[rng.rand(n)]
  words = text.split(' ')
  if words.size > 2
    words[rng.rand(words.size)] = "mut#{rng.rand(10_000)}"
  elsif !words.empty?
    words[0] = words[0] + "mut#{rng.rand(10_000)}"
  end
  words.join(' ')
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

puts "retention: path=#{path} train=#{n} test=#{test.size} classes=#{train.values.uniq.size} queries=#{query_corpus.size}"
puts "influence: total=#{profile.total_ms.round(1)}ms matmul=#{profile.matmul_ms.round(1)}ms"
puts
puts "%-8s %-10s %-12s %-12s" % ["K", "strategy", "decision", "escalation"]
ks = [10, 20, 40, 80, 120, 200, 300, 400, 500, 800, n].select { |k| k <= n }.uniq
ks.each do |k|
  strategies.each do |name, scores|
    keep = train_keys.sort_by { |key| -(scores[key]? || 0.0) }.first(k)
    keep_set = keep.to_set
    subset = Semrember::Memory.new(solver: :dual, min_df: 1).fit(train.select { |key, _| keep_set.includes?(key) })
    decision = 0
    escalation = 0
    test.each_with_index do |(q, _), idx|
      hit = subset.query(q, 1)[0]?
      decision += 1 if hit && hit[0] == full_pred[idx]
      escalation += 1 if (subset.novelty(q) > 0.55) == full_escalate[idx]
    end
    puts "%-8d %-10s %-12.3f %-12.3f" % [k, name, decision.to_f64 / test.size, escalation.to_f64 / test.size]
  end
  puts
end
