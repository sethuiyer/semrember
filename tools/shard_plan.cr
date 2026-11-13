# shard_plan.cr — offline shard-architecture planner.
#
#   crystal run --release tools/shard_plan.cr -- examples/incidents.csv \
#     --key message --value action --budget 20
#
# The planner builds label shards with a frozen feature map, measures top-1
# preservation, encodes deployment/assignment as weighted MaxSAT, solves with
# the Crystal NuWLS core, and trains a router on the resulting assignments.
# It does not change the service or write any state.
require "json"
require "../src/semrember"
require "../src/csv"
require "../src/shards"

path = nil.as(String?)
key_col = "message"
val_col = "action"
budget = nil.as(Int32?)
time_budget = 2.0
seed = 1_u64
fallback_cost = 10.0
args = ARGV.dup
i = 0
while i < args.size
  case args[i]
  when "--key"            then key_col = args[i + 1]; i += 2
  when "--value"          then val_col = args[i + 1]; i += 2
  when "--budget"         then budget = args[i + 1].to_i; i += 2
  when "--time"           then time_budget = args[i + 1].to_f; i += 2
  when "--seed"           then seed = args[i + 1].to_u64; i += 2
  when "--fallback-cost"  then fallback_cost = args[i + 1].to_f; i += 2
  when "--help", "-h"
    puts "usage: shard_plan <corpus.csv> [--key K] [--value V] [--budget N]"
    puts "                  [--time SECONDS] [--seed N] [--fallback-cost W]"
    puts ""
    puts "  Budget is retained observations across deployed label shards."
    puts "  The full model is assumed resident and is the fallback."
    exit 0
  else
    raise ArgumentError.new("unknown option #{args[i]}") if args[i].starts_with?('-')
    path = args[i]
    i += 1
  end
end
raise ArgumentError.new("missing corpus path") unless path

data = Semrember::Csv.load_pairs(path.not_nil!, key_col, val_col).to_h
raise ArgumentError.new("empty corpus") if data.empty?
parent = Semrember::Memory.new.fit(data)
queries = data.keys
budget = budget || [1, data.size // 2].max
plan = Semrember::Shards.build_plan(parent, queries, budget,
  time_budget: time_budget, seed: seed, fallback_cost: fallback_cost)

payload = JSON.build do |j|
  j.object do
    j.field "queries", queries.size
    j.field "candidates", plan.candidate_count
    j.field "budget", plan.budget
    j.field "budget_used", plan.budget_used
    j.field "selected" do
      j.array { plan.selected.each { |label| j.string label } }
    end
    j.field "fallback_count", plan.fallback_count
    j.field "router_accuracy", plan.router_accuracy
    j.field "valid", plan.valid
    j.field "solver_status", plan.solver_status
    j.field "solver_cost", plan.solver_cost
    j.field "assignments" do
      j.array do
        plan.assignments.each do |(query, destination)|
          j.object do
            j.field "query", query
            j.field "destination", destination
          end
        end
      end
    end
    j.field "scope", "top-1 fidelity; training-query router accuracy; fallback resident"
  end
end
puts payload
