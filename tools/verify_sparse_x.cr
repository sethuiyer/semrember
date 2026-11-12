require "../src/semrember"

# Small-scale verification of the "drop dense X" claim.
#
#   crystal run --release tools/verify_sparse_x.cr -- 1000
#
# It fits a real Memory, then compares the persistent dense X footprint against
# the sparse Features footprint, and times the two inner products that
# dual_insert needs (k = X x_new and X^T r).

private def sparse_dot(a : Semrember::Features, b : Semrember::Features) : Float64
  s = 0.0_f64
  ai = 0
  bi = 0
  while ai < a.idx.size && bi < b.idx.size
    av = a.idx[ai]
    bv = b.idx[bi]
    if av == bv
      s += a.val[ai].to_f64 * b.val[bi].to_f64
      ai += 1
      bi += 1
    elsif av < bv
      ai += 1
    else
      bi += 1
    end
  end
  s
end

private def dense_dot(a : Array(Float32), b : Array(Float32)) : Float64
  s = 0.0_f64
  a.each_with_index { |v, i| s += v.to_f64 * b[i].to_f64 }
  s
end

private def run(n : Int32)
  rng = Random.new(n * 31 + 7)
  words = Array(String).new(800) { |i| "w#{i}" }
  labels = Array(String).new(20) { |i| "label_#{i}" }
  corpus = {} of String => String
  n.times do |i|
    key = Array(String).new(14) { words[rng.rand(words.size)] }.join(' ')
    key += " uniq#{i}"
    corpus[key] = labels[rng.rand(labels.size)]
  end

  mem = Semrember::Memory.new(solver: :dual, min_df: 2).fit(corpus)
  keys = corpus.keys
  feats = keys.map { |k| mem.encode(k) }
  d = mem.d_tfidf
  nnz = feats.sum(&.idx.size)

  dense_bytes = n.to_i64 * d.to_i64 * 8
  sparse_bytes = nnz.to_i64 * 8 # Int32 index + Float32 value
  ratio = dense_bytes.to_f64 / sparse_bytes.to_f64

  puts "n=#{n} d=#{d} nnz=#{nnz} avg_nnz=#{(nnz.to_f64 / n).round(1)}"
  puts "  dense X:  #{dense_bytes // 1_048_576} MiB (N*D*8)"
  puts "  sparse X: #{sparse_bytes // 1_048_576} MiB (nnz*8), ratio #{ratio.round(1)}x"

  # Build the dense rows once; this is exactly the cache the proposal removes.
  dense_rows = feats.map { |f| f.dense(mem.d_tfidf, mem.d_static) }
  q = feats[0]
  q_dense = q.dense(mem.d_tfidf, mem.d_static)

  reps = 50
  sink = 0.0_f64

  t0 = Time.monotonic
  reps.times do
    dense_rows.each { |row| sink += dense_dot(row, q_dense) }
  end
  dense_k = (Time.monotonic - t0).total_milliseconds / reps

  t0 = Time.monotonic
  reps.times do
    feats.each { |f| sink += sparse_dot(f, q) }
  end
  sparse_k = (Time.monotonic - t0).total_milliseconds / reps

  # One X^T r product.  r is arbitrary but fixed.
  r = Array(Float64).new(n) { |i| (i % 7).to_f64 / 7.0 }
  acc = Array(Float64).new(d, 0.0_f64)
  reps = 20

  t0 = Time.monotonic
  reps.times do
    acc.fill(0.0_f64)
    d.times do |dd|
      s = 0.0_f64
      n.times { |i| s += dense_rows[i][dd].to_f64 * r[i] }
      acc[dd] = s
    end
    sink += acc[0]
  end
  dense_xtr = (Time.monotonic - t0).total_milliseconds / reps

  t0 = Time.monotonic
  reps.times do
    acc.fill(0.0_f64)
    n.times do |i|
      ri = r[i]
      next if ri == 0.0
      f = feats[i]
      f.idx.each_with_index { |ix, k| acc[ix] += f.val[k].to_f64 * ri }
    end
    sink += acc[0]
  end
  sparse_xtr = (Time.monotonic - t0).total_milliseconds / reps

  puts "  k = X x_new:  dense #{dense_k.round(2)} ms  sparse #{sparse_k.round(2)} ms  speedup #{(dense_k / sparse_k).round(1)}x"
  puts "  X^T r:        dense #{dense_xtr.round(2)} ms  sparse #{sparse_xtr.round(2)} ms  speedup #{(dense_xtr / sparse_xtr).round(1)}x"
  puts "  sink=#{sink.round(3)}"
end

sizes = ARGV.empty? ? [1000] : ARGV.map(&.to_i)
sizes.each { |n| run(n) }
