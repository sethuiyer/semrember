require "spec"
require "../src/shards"
require "../src/csv"

private def shard_data
  Semrember::Csv.load_pairs("examples/incidents.csv", "message", "action").to_h
end

describe Semrember::Shards do
  it "builds label candidates with frozen feature maps" do
    parent = Semrember::Memory.new.fit(shard_data)
    candidates = Semrember::Shards.label_candidates(parent)
    candidates.size.should eq parent.labels.size
    candidates.each do |candidate|
      candidate.memory.should be > 0
      candidate.model.labels.should eq parent.labels
      candidate.model.d_tfidf.should eq parent.d_tfidf
    end
  end

  it "produces a valid budgeted plan and a trainable router" do
    data = shard_data
    parent = Semrember::Memory.new.fit(data)
    queries = data.keys
    plan = Semrember::Shards.build_plan(parent, queries, 20,
      time_budget: 1.0, seed: 1_u64)

    plan.valid.should be_true
    plan.budget_used.should be <= 20
    plan.assignments.size.should eq queries.size
    plan.fallback_count.should be >= 0
    plan.router_accuracy.should be >= 0.9
    plan.selected.size.should be > 0
    plan.selected.each do |label|
      parent.labels.should contain label
    end
    (plan.selected.size + plan.fallback_count).should be > 0
  end

  it "respects a zero budget and a full budget" do
    data = shard_data
    parent = Semrember::Memory.new.fit(data)
    queries = data.keys

    zero = Semrember::Shards.build_plan(parent, queries, 0,
      time_budget: 1.0, seed: 2_u64)
    zero.valid.should be_true
    zero.selected.should be_empty
    zero.budget_used.should eq 0
    zero.fallback_count.should eq queries.size

    full = Semrember::Shards.build_plan(parent, queries, data.size,
      time_budget: 1.0, seed: 3_u64)
    full.valid.should be_true
    full.fallback_count.should eq 0
    full.budget_used.should eq data.size
    full.selected.size.should eq parent.labels.size
  end

  it "preserves top-1 for selected destinations" do
    data = shard_data
    parent = Semrember::Memory.new.fit(data)
    plan = Semrember::Shards.build_plan(parent, data.keys, 20,
      time_budget: 1.0, seed: 4_u64)
    plan.assignments.each do |(query, dest)|
      next unless dest.starts_with?("label:")
      expected = parent.query(query, 1)[0]?.try(&.[0])
      next unless expected
      label = dest.lchop("label:")
      child = parent.subset_by_labels([label])
      child.query(query, 1)[0]?.try(&.[0]).should eq expected
    end
  end
end
