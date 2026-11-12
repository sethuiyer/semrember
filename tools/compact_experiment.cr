require "../src/semrember"
require "../src/csv"
require "../src/influence"

# Counterfactual compaction on a real text corpus.
#
#   crystal run --release tools/compact_experiment.cr -- \
#     ~/Documents/Documents/Workspace/Datasets/BBC_News_Article_Train.csv \
#     --text-col text --label-col category --max-rows 1200 --queries 100

text_col = "text"
label_col = "category"
max_rows = 0
query_count = 100
coverage_floor = 0
seed = 23_u64
path : String? = nil
args = ARGV.dup
i = 0
while i < args.size
  case args[i]
  when "--text-col"       then text_col = args[i + 1]; i += 2
  when "--label-col"      then label_col = args[i + 1]; i += 2
  when "--max-rows"       then max_rows = args[i + 1].to_i; i += 2
  when "--queries"        then query_count = args[i + 1].to_i; i += 2
  when "--coverage-floor" then coverage_floor = args[i + 1].to_i; i += 2
  when "--seed"           then seed = args[i + 1].to_u64; i += 2
  else                         path = args[i]; i += 1
  end
end
raise "no corpus path" unless path

data = {} of String => String
Semrember::Csv.load_pairs(path, text_col, label_col).each { |(t, l)| data[t.downcase] = l }
if max_rows > 0 && data.size > max_rows
  data = data.to_a.shuffle(Random.new(seed)).first(max_rows).to_h
end
items = data.to_a.shuffle(Random.new(seed))
test_size = {1, items.size // 5}.max
test = items.first(test_size)
train = items[test_size..].to_h
train_keys = train.keys
n = train_keys.size
raise "corpus too small" if n < 20

full = Semrember::Memory.new(solver: :dual, min_df: 1).fit(train)
before_pred = test.map { |(q, _)| full.query(q, 1)[0]? ? full.query(q, 1)[0][0] : "" }

rng = Random.new(seed)
queries = Array(String).new(query_count) do
  words = train_keys[rng.rand(n)].split(' ')
  if words.size > 2
    words[rng.rand(words.size)] = "mut#{rng.rand(10_000)}"
  elsif !words.empty?
    words[0] = words[0] + "mut#{rng.rand(10_000)}"
  end
  words.join(' ')
end

t0 = Time.monotonic
result = full.compact(queries, beta_threshold: 0.0, coverage_floor: coverage_floor,
  keep_per_label: 1, verify: true)
ms = (Time.monotonic - t0).total_milliseconds

after_pred = test.map { |(q, _)| full.query(q, 1)[0]? ? full.query(q, 1)[0][0] : "" }
agreement = test.each_index.count { |idx| before_pred[idx] == after_pred[idx] }

puts "path=#{path} n=#{n} test=#{test.size} classes=#{train.values.uniq.size} queries=#{query_count} coverage_floor=#{coverage_floor}"
puts "compact: kept=#{result.kept} pruned=#{result.pruned} verified=#{result.verified} changed_queries=#{result.changed_queries} pinned=#{result.pinned_kept} label_kept=#{result.label_kept}"
puts "held-out decision agreement after compaction: #{agreement}/#{test.size} = #{(agreement.to_f64 / test.size).round(3)}"
puts "compact time: #{ms.round(0)}ms"
