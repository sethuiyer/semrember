require "spec"
require "../src/semrember"
require "../src/influence"

private def with_test_encoder(dim : Int32 = 2, &block : String ->)
  path = File.tempname("semrember-influence-encoder", ".bin")
  begin
    File.open(path, "wb") do |f|
      f << "M2VE"
      [1, 3, dim].each { |v| f.write(Semrember::Codec.u32(v)) }
      ["[UNK]", "disk", "host"].each do |token|
        f.write(Semrember::Codec.u16(token.bytesize))
        f << token
      end
      3.times do |i|
        dim.times { |d| f.write(Semrember::Codec.u16(Semrember::F16.from_f32(d == i % dim ? 1_f32 : 0_f32))) }
      end
    end
    yield path
  ensure
    File.delete(path) if File.exists?(path)
  end
end

private def random_corpus(rng : Random, n : Int32) : Hash(String, String)
  words = ["disk", "full", "host", "down", "cert", "expired", "dns", "slow",
           "cache", "stale", "memory", "oom", "network", "route", "service", "failed"]
  labels = ["free_disk", "fix_dns", "renew_certificate", "clear_cache", "increase_memory", "restart_service"]
  corpus = {} of String => String
  n.times do
    key = Array(String).new(4) { words[rng.rand(words.size)] }.join(' ')
    key += " #{rng.rand(1000)}" if rng.rand(3) == 0
    corpus[key] = labels[rng.rand(labels.size)]
  end
  corpus
end

private def random_queries(rng : Random, corpus : Hash(String, String), n : Int32) : Array(String)
  keys = corpus.keys
  Array(String).new(n) do
    if rng.rand(2) == 0
      keys[rng.rand(keys.size)]
    else
      k = keys[rng.rand(keys.size)].split(' ')
      k[rng.rand(k.size)] = "unknown#{rng.rand(100)}"
      k.join(' ')
    end
  end
end

private def assert_matches_exact(mem : Semrember::Memory, queries : Array(String))
  n = mem.stored
  queries.each do |q|
    n.times do |i|
      fast = mem.leave_one_out_scores(q, i)
      exact = mem.exact_leave_one_out_scores(q, i)
      fast.size.should eq exact.size
      fast.zip(exact).each { |a, b| a.should be_close(b, 1e-10) }
    end
  end
end

private def assert_banzhaf_matches_exact(mem : Semrember::Memory, queries : Array(String))
  results = mem.banzhaf_influence(queries, 2)
  n = mem.stored
  n.times do |i|
    pivotal = 0
    queries.each do |q|
      base = mem.scores64(q)
      alt = mem.exact_leave_one_out_scores(q, i)
      base_winner = (0...base.size).max_by { |c| base[c] }
      alt_winner = (0...alt.size).max_by { |c| alt[c] }
      pivotal += 1 if base_winner != alt_winner
    end
    results[i].pivotal_queries.should eq pivotal
    results[i].banzhaf.should be_close(pivotal.to_f64 / queries.size, 1e-12)
  end
end

describe "leave-one-out influence" do
  it "matches exact deletion on randomized memories and queries" do
    rng = Random.new(11)
    corpus = random_corpus(rng, 18)
    mem = Semrember::Memory.new.fit(corpus)
    queries = random_queries(rng, corpus, 12)
    assert_matches_exact(mem, queries)
    assert_banzhaf_matches_exact(mem, queries)
  end

  it "matches with weighted duplicate observations" do
    mem = Semrember::Memory.new.fit(
      {"disk full" => "free_disk", "host down" => "fix_dns", "bad cert" => "renew"},
      {"disk full" => 3},
    )
    queries = ["disk full", "host down", "cert bad", "unknown"]
    assert_matches_exact(mem, queries)
    assert_banzhaf_matches_exact(mem, queries)
  end

  it "matches with nearly identical features and different labels" do
    mem = Semrember::Memory.new.fit({
      "disk full" => "free_disk",
      "disk ful" => "clear_cache",
      "disk fullness" => "free_disk",
      "host down" => "fix_dns",
    })
    queries = ["disk full", "disk ful", "disk fullness", "host down", "disk"]
    assert_matches_exact(mem, queries)
    assert_banzhaf_matches_exact(mem, queries)
  end

  it "matches after insertion, deletion, and label overwrite" do
    rng = Random.new(23)
    corpus = random_corpus(rng, 14)
    mem = Semrember::Memory.new.fit(corpus)
    mem.set("inserted memory disk full", "free_disk")
    mem.set(corpus.keys.first, "clear_cache")
    mem.delete(corpus.keys.last)
    queries = random_queries(rng, corpus, 8) + ["inserted memory disk full"]
    assert_matches_exact(mem, queries)
    assert_banzhaf_matches_exact(mem, queries)
  end

  it "handles a single-label dataset" do
    mem = Semrember::Memory.new.fit({"disk full" => "same", "host down" => "same", "bad cert" => "same"})
    queries = ["disk full", "host down", "unknown"]
    assert_matches_exact(mem, queries)
    results = mem.banzhaf_influence(queries)
    results.each { |r| r.banzhaf.should eq 0.0 }
  end

  it "matches with static embeddings enabled" do
    with_test_encoder do |path|
      encoder = Semrember::StaticEncoder.new(path)
      mem = Semrember::Memory.new(static: encoder).fit({
        "disk" => "x", "host" => "y", "disk full" => "x", "host down" => "y",
      })
      queries = ["disk", "host", "disk full", "unknown"]
      assert_matches_exact(mem, queries)
      assert_banzhaf_matches_exact(mem, queries)
      mem.leverage_scores.size.should eq mem.stored
    end
  end

  it "reports empty results without queries" do
    mem = Semrember::Memory.new.fit({"disk full" => "x", "host down" => "y"})
    results = mem.banzhaf_influence([] of String)
    results.size.should eq 2
    results.each { |r| r.banzhaf.should eq 0.0 }
  end
end
