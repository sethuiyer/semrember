require "spec"
require "../src/semrember"
require "../src/influence"

private def index_data
  {
    "No space left on device"  => "free_disk",
    "filesystem is full"       => "free_disk",
    "disk quota exceeded"      => "free_disk",
    "certificate has expired"  => "renew_certificate",
    "tls handshake failed"     => "renew_certificate",
    "x509 expired"             => "renew_certificate",
    "host down"                => "fix_dns",
    "dns timeout"              => "fix_dns",
    "cannot resolve host"      => "fix_dns",
    "cache stale"              => "clear_cache",
    "stale cache entry"        => "clear_cache",
    "cache corruption"         => "clear_cache",
  }
end

private def assert_indexed_query_matches(mem : Semrember::Memory, queries : Array(String))
  queries.each do |q|
    full = mem.scores(q)
    expected_order = (0...full.size).to_a.sort_by { |i| {-full[i], i} }
    [1, 2, 3].each do |k|
      next if k >= mem.labels.size
      expected = expected_order.first(k)
      hits = mem.query(q, k)
      hits.map(&.[0]).should eq(expected.map { |i| mem.labels[i] })
      hits.each_with_index do |(_, score), j|
        score.unsafe_as(UInt32).should eq(full[expected[j]].unsafe_as(UInt32))
      end
    end
  end
end

{% if flag?(:linux) %}
  describe "Memory weight-index integration" do
    it "uses a rebuilt index for exact top-k parity and bitwise finalist scores" do
      mem = Semrember::Memory.new(index: Semrember::Mips::Config.new(rank: 3, leaf_size: 2))
        .fit(index_data)
      mem.index_state.should eq(:stale)
      mem.rebuild_index!.should be_true
      mem.index_ready?.should be_true
      mem.index_version.should_not be_nil

      queries = [
        "No space left on device",
        "certificate has expired",
        "host down",
        "cache stale",
        "disk full",
        "unknown service failure",
      ]
      assert_indexed_query_matches(mem, queries)
      mem.query("disk full", 0).should be_empty
      # topk == label count deliberately takes the full-scan path.
      full_hits = mem.query("disk full", mem.labels.size)
      expected = (0...mem.scores("disk full").size).to_a
        .sort_by { |i| {-mem.scores("disk full")[i], i} }
      full_hits.map(&.[0]).should eq(expected.map { |i| mem.labels[i] })
    end

    it "invalidates on writes and deletes, then rebuilds with parity" do
      mem = Semrember::Memory.new(index: Semrember::Mips::Config.new(rank: 3, leaf_size: 2))
        .fit(index_data)
      mem.rebuild_index!.should be_true
      mem.index_ready?.should be_true

      mem.set("new incident", "free_disk")
      mem.index_ready?.should be_false
      mem.index_state.should eq(:stale)
      assert_indexed_query_matches(mem, ["new incident", "certificate has expired"])
      mem.rebuild_index!.should be_true
      mem.index_ready?.should be_true
      assert_indexed_query_matches(mem, ["new incident", "certificate has expired"])

      mem.delete("new incident").should be_true
      mem.index_ready?.should be_false
      assert_indexed_query_matches(mem, ["No space left on device"])
      mem.rebuild_index!.should be_true
      assert_indexed_query_matches(mem, ["No space left on device"])
    end

    it "builds a reader index from AMEM and keeps query parity" do
      path = File.tempname("semrember-index", ".amem")
      begin
        Semrember::Memory.new.fit(index_data).save(path)
        reader = Semrember::Memory.load(path)
        reader.index_state.should eq(:disabled)
        reader.enable_index(Semrember::Mips::Config.new(rank: 3, leaf_size: 2))
        reader.rebuild_index!.should be_true
        reader.index_ready?.should be_true
        assert_indexed_query_matches(reader, ["certificate has expired", "host down", "cache stale"])
        reader.disable_index
        reader.index_state.should eq(:disabled)
        assert_indexed_query_matches(reader, ["certificate has expired"])
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    it "does not share an index with an influence copy" do
      mem = Semrember::Memory.new(index: Semrember::Mips::Config.new(rank: 2, leaf_size: 2))
        .fit(index_data)
      mem.rebuild_index!.should be_true
      copy = mem.duplicate_for_influence
      copy.index_ready?.should be_false
      copy.index_version.should be_nil
      assert_indexed_query_matches(copy, ["certificate has expired", "cache stale"])
    end
  end
{% end %}
