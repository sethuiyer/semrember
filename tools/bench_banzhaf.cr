require "../src/semrember"
require "../src/influence"

# Benchmark the exact leave-one-out Banzhaf/influence pipeline.
#
#   crystal run --release -Dlapack tools/bench_banzhaf.cr -- 1000 2000 4000 6000

def corpus_for(n : Int32) : Hash(String, String)
  rng = Random.new(n * 31 + 7)
  words = Array(String).new(800) { |i| "w#{i}" }
  labels = Array(String).new(20) { |i| "label_#{i}" }
  corpus = {} of String => String
  n.times do |i|
    key = Array(String).new(14) { words[rng.rand(words.size)] }.join(' ')
    key += " uniq#{i}"
    corpus[key] = labels[rng.rand(labels.size)]
  end
  corpus
end

def queries_for(corpus : Hash(String, String), count : Int32) : Array(String)
  rng = Random.new(corpus.size * 17 + 3)
  keys = corpus.keys
  Array(String).new(count) do |i|
    if i % 3 == 0
      k = keys[rng.rand(keys.size)].split(' ')
      k[rng.rand(k.size)] = "mut#{rng.rand(10_000)}"
      k.join(' ')
    else
      keys[rng.rand(keys.size)]
    end
  end
end

def run(n : Int32, q_count : Int32 = 200)
  corpus = corpus_for(n)
  queries = queries_for(corpus, q_count)

  t0 = Time.monotonic
  mem = Semrember::Memory.new(solver: :dual, min_df: 2).fit(corpus)
  fit_ms = (Time.monotonic - t0).total_milliseconds

  profile = Semrember::BanzhafProfile.new
  t0 = Time.monotonic
  results = mem.banzhaf_influence(queries, 0, profile)
  total_ms = (Time.monotonic - t0).total_milliseconds

  # Naive exact-delete-and-query.  Full mode is only practical for small N.
  keys = mem.memory_keys
  full_naive = ARGV.includes?("--full-naive")
  naive_sample_ms = 0.0
  sample = 0
  if full_naive
    t0 = Time.monotonic
    n.times do |i|
      copy = mem.duplicate_for_influence
      copy.delete(keys[i])
      queries.each { |q| copy.query(q, 1) }
    end
    naive_full_ms = (Time.monotonic - t0).total_milliseconds
    naive_per_memory_ms = naive_full_ms / n
    naive_est_ms = naive_full_ms
  else
    sample = {10, n}.min
    t0 = Time.monotonic
    sample.times do |i|
      copy = mem.duplicate_for_influence
      copy.delete(keys[i])
      queries.each { |q| copy.query(q, 1) }
    end
    naive_sample_ms = (Time.monotonic - t0).total_milliseconds
    naive_per_memory_ms = naive_sample_ms / sample
    naive_est_ms = naive_per_memory_ms * n
    naive_full_ms = 0.0
  end

  t = mem.label_count
  a_mb = n.to_f64 * n.to_f64 * 8 / 1024.0 / 1024.0
  k_mb = n.to_f64 * q_count.to_f64 * 8 / 1024.0 / 1024.0
  z_mb = k_mb
  b_mb = n.to_f64 * t.to_f64 * 8 / 1024.0 / 1024.0

  puts "n=#{n} d=#{mem.d_tfidf} t=#{t} q=#{q_count} fit=#{fit_ms.round(0)}ms"
  puts "  phases: encode=#{profile.encode_ms.round(1)} kernel=#{profile.kernel_ms.round(1)} " \
       "precompute=#{profile.precompute_ms.round(1)} matmul=#{profile.matmul_ms.round(1)} " \
       "scores=#{profile.scores_ms.round(1)} eval=#{profile.eval_ms.round(1)} total=#{profile.total_ms.round(1)}ms"
  puts "  throughput: #{(q_count / (total_ms / 1000.0)).round(1)} q/s, " \
       "#{(n / (total_ms / 1000.0)).round(1)} memory/s, " \
       "#{((n.to_f64 * q_count) / (total_ms / 1000.0) / 1e6).round(2)} M pairs/s"
  if full_naive
    puts "  naive delete+query: FULL #{naive_full_ms.round(0)}ms, " \
         "per-memory #{naive_per_memory_ms.round(2)}ms " \
         "(speedup #{(naive_est_ms / total_ms).round(1)}x)"
  else
    puts "  naive delete+query: sample #{sample} = #{naive_sample_ms.round(0)}ms, " \
         "per-memory #{naive_per_memory_ms.round(2)}ms, est total #{naive_est_ms.round(0)}ms " \
         "(speedup #{(naive_est_ms / total_ms).round(1)}x)"
  end
  puts "  dense arrays: A=#{a_mb.round(1)}MiB K=#{k_mb.round(1)}MiB Z=#{z_mb.round(1)}MiB B=#{b_mb.round(1)}MiB"
  puts "  top mean influence: #{results.sort_by { |r| -r.mean_influence }.first(3).map { |r| "#{r.key}=#{r.mean_influence.round(3)}" }.join(", ")}"
end

sizes = ARGV.reject(&.starts_with?("--")).map(&.to_i)
sizes = [1000] if sizes.empty?
sizes.each { |n| run(n) }
