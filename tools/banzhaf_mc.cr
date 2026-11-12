require "../src/semrember"
require "../src/influence"

# Classical coalition Banzhaf vs Monte Carlo, on a small memory set.
#
#   crystal run --release tools/banzhaf_mc.cr
#
# The leave-one-out beta in src/influence.cr is a different quantity.  This
# tool clearly names v(S) as the decision vector over a coalition S and
# estimates the classical coalition Banzhaf index.

struct BanzhafEstimate
  getter key : String
  getter estimate : Float64
  getter lower : Float64
  getter upper : Float64
  getter samples : Int32

  def initialize(@key, @estimate, @lower, @upper, @samples)
  end
end

def decisions_for_mask(mem : Semrember::Memory, keys : Array(String), mask : UInt64,
                       queries : Array(String)) : Array(String)
  return Array(String).new(queries.size, "∅") if mask == 0
  copy = mem.duplicate_for_influence
  remove = [] of String
  keys.each_with_index do |key, i|
    remove << key unless (mask & (1_u64 << i)) != 0
  end
  copy.delete_many(remove) unless remove.empty?
  queries.map do |q|
    scores = copy.scores64(q)
    winner = 0
    best = scores[0]
    (1...scores.size).each do |c|
      if scores[c] > best
        best = scores[c]
        winner = c
      end
    end
    copy.labels[winner]
  end
end

def exact_banzhaf(mem : Semrember::Memory, keys : Array(String), queries : Array(String)) : Array(Float64)
  n = keys.size
  raise "exact enumeration is only for n <= 12" if n > 12
  cache = Hash(UInt64, Array(String)).new
  decisions = ->(mask : UInt64) { cache[mask] ||= decisions_for_mask(mem, keys, mask, queries) }
  beta = Array(Float64).new(n, 0.0)
  n.times do |i|
    pivotal = 0
    trials = 0
    (0_u64...2_u64 ** n).each do |mask|
      next if (mask & (1_u64 << i)) != 0
      a = decisions.call(mask)
      b = decisions.call(mask | (1_u64 << i))
      queries.size.times { |qq| pivotal += 1 if a[qq] != b[qq] }
      trials += queries.size
    end
    beta[i] = pivotal.to_f64 / trials
  end
  beta
end

def mc_banzhaf(mem : Semrember::Memory, keys : Array(String), queries : Array(String),
               max_samples : Int32 = 400, min_samples : Int32 = 40,
               target_half_width : Float64 = 0.03, seed : UInt64 = 7) : Array(BanzhafEstimate)
  n = keys.size
  rng = Random.new(seed)
  cache = Hash(UInt64, Array(String)).new
  decisions = ->(mask : UInt64) { cache[mask] ||= decisions_for_mask(mem, keys, mask, queries) }
  out = [] of BanzhafEstimate
  n.times do |i|
    sum = 0.0
    sum2 = 0.0
    m = 0
    p = 0.0
    half = 1.0
    while m < max_samples
      mask = 0_u64
      n.times { |j| mask |= (1_u64 << j) if rng.rand(2) == 1 }
      without_i = mask & ~(1_u64 << i)
      with_i = without_i | (1_u64 << i)
      a = decisions.call(without_i)
      b = decisions.call(with_i)
      diff = 0
      queries.size.times { |qq| diff += 1 if a[qq] != b[qq] }
      frac = diff.to_f64 / queries.size
      sum += frac
      sum2 += frac * frac
      m += 1
      p = sum / m
      if m > 1
        var = (sum2 / m - p * p) * m / (m - 1)
        half = 1.96 * Math.sqrt(var / m)
      end
      break if m >= min_samples && half <= target_half_width
    end
    out << BanzhafEstimate.new(keys[i], p, (p - half).clamp(0.0, 1.0),
      (p + half).clamp(0.0, 1.0), m)
  end
  out
end

mem = Semrember::Memory.new.fit({
  "disk full" => "free_disk",
  "no space left" => "free_disk",
  "host down" => "fix_dns",
  "dns timeout" => "fix_dns",
  "cert expired" => "renew_certificate",
  "tls handshake failed" => "renew_certificate",
  "cache stale" => "clear_cache",
  "oom killed" => "increase_memory",
})
keys = mem.memory_keys
queries = ["disk full", "no space left", "host down", "cert expired", "cache stale", "unknown failure"]
exact = exact_banzhaf(mem, keys, queries)
mc = mc_banzhaf(mem, keys, queries)
puts "%-22s %8s %8s %-18s %8s" % ["memory", "exact", "mc", "95% CI", "samples"]
mc.each_with_index do |e, i|
  puts "%-22s %8.3f %8.3f [%6.3f, %6.3f] %8d" % [e.key, exact[i], e.estimate, e.lower, e.upper, e.samples]
end
