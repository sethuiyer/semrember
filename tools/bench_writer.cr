require "../src/semrember"

# Synthetic writer benchmark: sparse lexical features, no static encoder.
# Measures fit, first novelty, insert, delete, and bulk delete.
struct WriterBench
  def self.run(n : Int32, min_df : Int32 = 2)
    rng = Random.new(n * 17 + min_df)
    words = Array(String).new(800) { |i| "w#{i}" }
    labels = Array(String).new(20) { |i| "label_#{i}" }
    corpus = {} of String => String
    n.times do |i|
      key = Array(String).new(14) { words[rng.rand(words.size)] }.join(' ')
      key += " uniq#{i}"
      corpus[key] = labels[rng.rand(labels.size)]
    end

    t0 = Time.monotonic
    mem = Semrember::Memory.new(solver: :dual, min_df: min_df).fit(corpus)
    fit_ms = (Time.monotonic - t0).total_milliseconds

    sample = corpus.keys.first
    t0 = Time.monotonic
    mem.novelty(sample)
    novelty_ms = (Time.monotonic - t0).total_milliseconds

    new_key = "inserted key w1 w2 w3 uniq_insert"
    t0 = Time.monotonic
    mem.set(new_key, labels[0])
    insert_ms = (Time.monotonic - t0).total_milliseconds

    t0 = Time.monotonic
    mem.delete(new_key)
    delete_ms = (Time.monotonic - t0).total_milliseconds

    keys = corpus.keys.first(20)
    t0 = Time.monotonic
    mem.delete_many(keys)
    bulk_ms = (Time.monotonic - t0).total_milliseconds

    puts "n=#{n} min_df=#{min_df} d=#{mem.d_tfidf} labels=#{mem.label_count} " \
         "fit=#{fit_ms.round(1)}ms novelty=#{novelty_ms.round(1)}ms " \
         "insert=#{insert_ms.round(2)}ms delete=#{delete_ms.round(2)}ms " \
         "bulk20=#{bulk_ms.round(1)}ms"
  end
end

sizes = ARGV.empty? ? [100, 300, 500, 850] : ARGV.map(&.to_i)
sizes.each { |n| WriterBench.run(n) }
