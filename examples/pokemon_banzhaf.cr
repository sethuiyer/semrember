require "../src/semrember"
require "../src/csv"
require "../src/influence"

# Banzhaf-style leave-one-out influence on the Pokémon name -> primary type
# cache, after one live correction.
#
#   crystal run --release examples/pokemon_banzhaf.cr -- ~/semrember/build/pokemon.csv

path = ARGV[0]? || File.expand_path("~/semrember/build/pokemon.csv")
data = {} of String => String
Semrember::Csv.load_pairs(path, "Name", "Type 1").each do |(name, type)|
  data[name.downcase] = type
end
raise "empty Pokemon corpus" if data.empty?

mem = Semrember::Memory.new(solver: :dual, min_df: 1).fit(data)
mem.set("mooshikachu", "Normal")
keys = mem.memory_keys
idx = keys.index("mooshikachu").not_nil!

queries = [
  "mooshikachu", "mooshikachoo", "mooshichu", "mooshikachu-x",
  "mooshiraichu", "pikachu", "raichu", "pichu", "smoochum",
]

profile = Semrember::BanzhafProfile.new
results = mem.banzhaf_influence(queries, 5, profile)
by_key = results.to_h { |r| {r.key, r} }

puts "pokemon: n=#{mem.stored} d=#{mem.d_tfidf} labels=#{mem.label_count} q=#{queries.size}"
puts "influence phases: encode=#{profile.encode_ms.round(1)}ms kernel=#{profile.kernel_ms.round(1)}ms " \
     "precompute=#{profile.precompute_ms.round(1)}ms matmul=#{profile.matmul_ms.round(1)}ms " \
     "scores=#{profile.scores_ms.round(1)}ms eval=#{profile.eval_ms.round(1)}ms total=#{profile.total_ms.round(1)}ms"
puts

puts "%-16s %-8s %-8s %-8s %s" % ["memory", "beta", "pivotal", "mean-I", "examples"]
["mooshikachu", "pikachu", "raichu", "pichu", "smoochum"].each do |key|
  r = by_key[key]?
  next unless r
  puts "%-16s %-8.3f %-8d %-8.3f %s" % [key, r.banzhaf, r.pivotal_queries, r.mean_influence, r.examples.join(", ")]
end

puts
puts "top by mean influence:"
results.sort_by { |r| -r.mean_influence }.first(6).each do |r|
  puts "  %-16s beta=%.3f mean=%.3f max=%.3f" % [r.key, r.banzhaf, r.mean_influence, r.max_influence]
end

puts
puts "mooshikachu counterfactual:"
copy = mem.duplicate_for_influence
copy.delete("mooshikachu")
queries.each do |q|
  base = mem.query(q, 1)[0]? ? mem.query(q, 1)[0][0] : "-"
  fast = mem.leave_one_out_prediction(q, idx)
  exact = copy.query(q, 1)[0]? ? copy.query(q, 1)[0][0] : "-"
  fast_scores = mem.leave_one_out_scores(q, idx)
  exact_scores = mem.exact_leave_one_out_scores(q, idx)
  err = fast_scores.zip(exact_scores).map { |a, b| (a - b).abs }.max
  puts "  %-14s base=%-12s without=%-12s exact=%-12s score_err=%.2e" % [q, base, fast, exact, err]
end
