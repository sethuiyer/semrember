require "spec"
require "../src/semrember"

private def subset_data
  {
    "No space left on device"  => "free_disk",
    "filesystem is full"       => "free_disk",
    "disk quota exceeded"      => "free_disk",
    "certificate has expired"  => "renew_certificate",
    "tls handshake failed"     => "renew_certificate",
    "host down"                => "fix_dns",
    "dns timeout"              => "fix_dns",
    "cache stale"              => "clear_cache",
  }
end

describe Semrember::Memory do
  it "builds a shard with the parent's frozen feature map and full labels" do
    parent = Semrember::Memory.new.fit(subset_data)
    child = parent.subset_by_labels(["free_disk"])

    child.labels.should eq parent.labels
    child.d_tfidf.should eq parent.d_tfidf
    child.d_static.should eq parent.d_static
    child.stored.should eq 3
    child.writable?.should be_true

    key = "No space left on device"
    child.encode(key).idx.should eq parent.encode(key).idx
    child.encode(key).val.should eq parent.encode(key).val

    scores = child.scores("No space left on device")
    scores.size.should eq parent.labels.size
    free = parent.labels.index("free_disk").not_nil!
    renew = parent.labels.index("renew_certificate").not_nil!
    scores[free].should be > 0.0_f32
    scores[renew].should eq 0.0_f32
    child.writer_residual.should be < 1e-5
  end

  it "preserves observation order and index-selected features" do
    parent = Semrember::Memory.new.fit(subset_data)
    child = parent.subset([0, 4, 6])
    child.stored.should eq 3
    child.labels.should eq parent.labels
    child.writer_residual.should be < 1e-5
    child.scores("certificate has expired").size.should eq parent.labels.size

    empty = parent.subset([] of Int32)
    empty.stored.should eq 0
    empty.scores("anything").all? { |v| v == 0.0_f32 }.should be_true
  end

  it "does not share mutable model state with the parent" do
    parent = Semrember::Memory.new.fit(subset_data)
    child = parent.subset_by_labels(["free_disk"])
    before = child.scores("No space left on device")
    before_labels = child.labels.dup
    parent.set("a new key", "new_label")
    child.labels.should eq before_labels
    parent.labels.should_not eq child.labels
    child.scores("No space left on device").should eq before
  end

  it "rejects out-of-range indices and non-dual parents" do
    parent = Semrember::Memory.new.fit(subset_data)
    expect_raises(ArgumentError) { parent.subset([999]) }
    delta = Semrember::Memory.new(solver: :delta).fit(subset_data)
    expect_raises(Semrember::UnavailableError) { delta.subset([0]) }
  end
end
