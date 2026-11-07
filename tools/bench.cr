require "../src/semrember"

# Microbenchmark: in-process query latency of the Crystal engine.
#
#   crystal run --release tools/bench.cr model.amem [n]

model = ENV["AMEM"]? || "/tmp/srv.amem"
n = (ENV["N"]? || "100000").to_i

mem = Semrember::Memory.load(model)
1000.times { mem.query("No space left on device", 1) }

t0 = Time.monotonic
n.times { mem.query("No space left on device", 1) }
d = Time.monotonic - t0

us = d.nanoseconds.to_f64 / n / 1000.0
puts "crystal query: #{us.round(3)} us  (n=#{n}, labels=#{mem.label_count}, d=#{mem.d_tfidf})"
