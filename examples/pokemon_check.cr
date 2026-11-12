require "../src/semrember"
require "../src/csv"

# Pokémon name -> primary type, using the CSV from the mlhub Pokémon corpus.
#
#   crystal run --release examples/pokemon_check.cr -- ~/semrember/build/pokemon.csv
#
# This is a real use-case check, not a benchmark: exact known names, held-out
# spelling transfer, novelty, similarity, and one live correction.

path = ARGV[0]? || File.expand_path("~/semrember/build/pokemon.csv")
pairs = Semrember::Csv.load_pairs(path, "Name", "Type 1")
data = {} of String => String
pairs.each { |(name, type)| data[name.downcase] = type }
raise "empty Pokemon corpus" if data.empty?

items = data.to_a
items.shuffle(Random.new(0))
test_size = {1, items.size // 5}.max
test = items.first(test_size)
train = items[test_size..]
train_data = train.to_h

t0 = Time.monotonic
train_mem = Semrember::Memory.new(solver: :dual, min_df: 1).fit(train_data)
train_ms = (Time.monotonic - t0).total_milliseconds

top1 = 0
top2 = 0
test.each do |(name, expected)|
  hits = train_mem.query(name, 2)
  top1 += 1 if hits[0]? && hits[0][0] == expected
  top2 += 1 if hits.any? { |(label, _)| label == expected }
end

t0 = Time.monotonic
mem = Semrember::Memory.new(solver: :dual, min_df: 1).fit(data)
fit_ms = (Time.monotonic - t0).total_milliseconds
mem.novelty("pikachu") # cache the novelty inverse before live writes

puts "pokemon: n=#{data.size} d=#{mem.d_tfidf} labels=#{mem.label_count}"
puts "held-out: #{top1}/#{test_size} top-1, #{top2}/#{test_size} top-2"
puts "fit: held-out #{train_ms.round(1)} ms, full #{fit_ms.round(1)} ms"

queries = [
  "pikachu", "charizard", "bulbasaur", "mewtwo", "diancie", "volcanion",
  "zapchu", "pikachoo", "charmanderr", "squirtel", "bulbasor",
  "mr mime", "aegislash", "dragapult", "flamebird", "agumon",
]

puts
puts "%-14s %-10s %-10s %-10s %-8s %s" % ["query", "top1", "top2", "top3", "novelty", "nearest"]
queries.each do |q|
  hits = mem.query(q, 3)
  novelty = mem.novelty(q)
  nearest = mem.similar(q, 1)[0]?
  puts "%-14s %-10s %-10s %-10s %-8.3f %s" % [
    q,
    hits[0]? ? "#{hits[0][0]} #{hits[0][1].round(2)}" : "-",
    hits[1]? ? "#{hits[1][0]} #{hits[1][1].round(2)}" : "-",
    hits[2]? ? "#{hits[2][0]} #{hits[2][1].round(2)}" : "-",
    novelty,
    nearest ? nearest[0] : "-",
  ]
end

puts
puts "before write:"
["mooshikachu", "mooshikachoo", "mooshichu", "pikachu", "raichu"].each do |q|
  hit = mem.query(q, 1)[0]?
  puts "  %-14s %s" % [q, hit ? "#{hit[0]} #{hit[1].round(3)}" : "-"]
end
mem.set("mooshikachu", "Normal")
puts "after set(mooshikachu -> Normal):"
["mooshikachu", "mooshikachoo", "mooshichu", "pikachu", "raichu"].each do |q|
  hit = mem.query(q, 1)[0]?
  puts "  %-14s %s" % [q, hit ? "#{hit[0]} #{hit[1].round(3)}" : "-"]
end
puts "writer residual: #{mem.writer_residual}"
