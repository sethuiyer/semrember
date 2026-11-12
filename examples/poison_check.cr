require "../src/semrember"
require "../src/influence"

# Counterfactual poisoning check on a small fixed corpus.
#
#   crystal run --release examples/poison_check.cr
#
# The check is a model-sensitivity signal, not proof of malicious intent.

rows = [
  {"disk full", "free_disk"}, {"no space left", "free_disk"}, {"filesystem full", "free_disk"},
  {"host down", "fix_dns"}, {"dns timeout", "fix_dns"}, {"cannot resolve host", "fix_dns"},
  {"cert expired", "renew_cert"}, {"tls handshake failed", "renew_cert"}, {"x509 expired", "renew_cert"},
  {"cache stale", "clear_cache"}, {"stale cache entry", "clear_cache"}, {"cache corruption", "clear_cache"},
  {"disk full on host", "free_disk"}, {"dns cache stale", "fix_dns"}, {"cert cache expired", "renew_cert"},
]
data = {} of String => String
rows.each { |(k, v)| data[k] = v }
mem = Semrember::Memory.new(solver: :dual, min_df: 1, rls_reg: 0.05).fit(data)
queries = ["disk full", "cert expired", "cache stale", "dns timeout", "unknown"]

[
  {"cache stale variant", "clear_cache"},
  {"oom network slow 33", "free_disk"},
  {"quota network expired 16", "free_disk"},
].each do |(key, label)|
  result = mem.poison_check(key, label, queries)
  puts "#{key.inspect} -> #{label}"
  puts "  suspicion=#{result.suspicion.round(3)} novelty=#{result.novelty.round(3)} " \
       "pivotal=#{result.pivotal_queries} dissimilar=#{result.dissimilar_pivots} " \
       "verdict=#{result.verdict} a_ii=#{result.a_ii.round(4)}"
  result.flipped.each do |f|
    puts "  #{f.query.inspect}: #{f.old_label} -> #{f.new_label} cosine=#{f.cosine.round(3)}"
  end
end
