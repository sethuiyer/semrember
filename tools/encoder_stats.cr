require "../src/static_encoder"

# Report load time and resident memory for a static encoder.
#
#   MODEL=models/potion-multilingual-128M.bin crystal run --release tools/encoder_stats.cr
#
# The Crystal encoder reads the model into memory; the C implementation
# memory-maps it.  For the 265 MB multilingual model that difference is
# visible in resident memory, so it is worth measuring rather than assuming.

path = ENV["MODEL"]? || "models/potion-multilingual-128M.bin"

t0 = Time.monotonic
enc = Semrember::StaticEncoder.new(path)
load = (Time.monotonic - t0).total_seconds

rss_kb = `ps -o rss= -p #{Process.pid}`.to_i64
file_mb = File.size(path) / 1048576.0

puts "model   : #{File.basename(path)}"
puts "load    : #{load.round(2)} s"
puts "file    : #{file_mb.round(1)} MB"
puts "rss     : #{(rss_kb / 1024.0).round(1)} MB"
puts "vocab   : #{enc.vocab_size}"
puts "dim     : #{enc.dim}"
puts "tokenizer: #{enc.tokenizer}"
