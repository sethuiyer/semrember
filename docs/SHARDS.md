# SUTRA shard planner

[README](../README.md) · [MaxSAT core](MAXSAT.md) · [Read-side weight index](MIPS.md) · [Influence](INFLUENCE.md)

`tools/shard_plan.cr` is an offline planner for a first hierarchical shard
architecture. It uses Semrember for the evidence and the Crystal NuWLS core for
the combinatorial choice:

```
frozen-feature shard candidates
  → H(q, s): does shard s preserve the full model's top-1?
  → weighted MaxSAT: deploy shards + assign queries under a memory budget
  → greedy feasible warm start
  → NuWLS solve
  → router trained on the selected assignments
```

It is deliberately a conservative first slice: **top-1 fidelity**, **label
shards**, and a **full-model fallback that is assumed resident**. It does not
change the service, write state, or claim an overall speedup.

## 1. Exact evidence: frozen-feature shards

`Memory#subset(indices)` and `Memory#subset_by_labels(labels)` build a dual
model over selected observations while reusing the parent's vocabulary, IDF,
label coordinates, static encoder, and ridge penalty. The feature map is never
refit, so the shard and parent scores live in the same space.

For every query `q` and candidate `s`:

```
H(q, s) = 1  iff  argmax_c s_c(q; shard_s) == argmax_c s_c(q; full)
```

The implementation evaluates this directly. The candidate model keeps all `T`
label coordinates; labels absent from the subset have zero rows in `W`, which
matches the group-deletion identity. The exact group formula remains available
for analysis; the planner uses direct shard evaluation for clarity.

## 2. Weighted MaxSAT encoding

Variables:

| variable | meaning |
|---|---|
| `a_s` | shard `s` is deployed |
| `r_{q,s}` | query `q` is assigned to shard `s` |
| `r_{q,0}` | query `q` uses the full-model fallback |
| `u_{s,j}` | unary memory unit for deployed shard `s` |
| counter aux | sequential at-most-budget encoding |

Hard clauses:

- exactly one destination per query (`r_{q,0}` or one `r_{q,s}`);
- `r_{q,s} → a_s`;
- `¬r_{q,s}` whenever `H(q,s) = 0`;
- `u_{s,j} ↔ a_s`, so a deployed shard contributes exactly `m_s` units;
- `Σ m_s a_s ≤ B` via a Sinz sequential counter.

Soft clauses:

- `¬r_{q,s}` weighted by the per-query shard cost;
- `¬r_{q,0}` weighted by the fallback cost;
- `¬a_s` weighted by `m_s`.

The full-model fallback is always available and is not charged against the
shard budget; it represents the already-resident parent model. Budget is
measured in **retained observations**, not bytes, so the objective is a
coverage/memory tradeoff rather than a latency claim.

## 3. Warm start

A pure NuWLS run from an all-fallback assignment can stall: moving a query to a
shard and deploying that shard are separate flips. The planner therefore builds
a greedy feasible assignment first:

- coverage of each candidate (`H = 1` count);
- benefit `(fallback_cost - label_cost) · cover − memory`;
- deploy affordable candidates in descending benefit;
- assign each query to a selected shard when possible, otherwise fallback.

The full assignment, including the at-most counter auxiliaries, is passed as
`warm_start`. The solver registers it as the incumbent and can only improve it.

## 4. Router

The selected `q → destination` assignments become router targets. A compact
`Memory` is fit on `query text → destination`, then evaluated on the same
queries. The planner reports **training-query router accuracy**, not held-out
accuracy. A held-out split is required before any deployment claim.

## 5. Example

```bash
make shard-plan
# or:
crystal run --release tools/shard_plan.cr -- examples/incidents.csv \
  --key message --value action --budget 20 --time 2 --seed 1
```

Current bundled corpus (34 queries, 10 label candidates, budget 20 observations):

```json
{"queries":34,"candidates":10,"budget":20,"budget_used":19,
 "selected":["free_disk","restart_service","renew_certificate","fix_dns","restart_network"],
 "fallback_count":15,"router_accuracy":1.0,"valid":true,
 "solver_status":"FEASIBLE","solver_cost":188.0}
```

Cost arithmetic: 19 retained observations + 19 label assignments + 15 fallbacks
at weight 10 = **188**. The full fallback alone costs 340.

Budget sweep:

| budget | selected | used | fallbacks | solver cost | valid |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 34 | 340 | yes |
| 4 | 1 | 4 | 30 | 308 | yes |
| 6 | 1 | 5 | 29 | 300 | yes |
| 10 | 2 | 9 | 25 | 268 | yes |
| 20 | 5 | 19 | 15 | 188 | yes |
| 34 | 10 | 34 | 0 | 68 | yes |

## 6. Limits

- **Top-1 only.** Escalation/top-two fidelity is not in `H`; a shard can
  preserve the winner while changing the margin or the escalation decision.
- **Label shards only.** Observation-level clusters, overlapping shards, and
  union experts are not generated yet.
- **Fallback is resident and free.** The full model's memory is not in the
  budget; shard memory is additional, not a replacement.
- **Router accuracy is in-sample.** No held-out query split yet.
- **NuWLS is an approximator.** `valid` means the returned assignment satisfies
  the encoded hard clauses; it is not a proof of optimality.
- **Budget is observations.** No byte-level A/W accounting, feature rows, or
  router state.
- **No service integration.** The planner is offline; the server still uses one
  model and the optional weight index.

## 7. Next slices

1. Add top-two / escalation fidelity to `H`.
2. Add observation-cluster and overlapping candidates.
3. Include A/W/router memory and measured per-shard cost.
4. Held-out router evaluation and a feedback loop from router confusion.
5. Enable the read-side weight index per shard and measure the combined cost.
6. Union/merge scoring for top-k probes.

## 8. Source map

| Piece | Code |
|---|---|
| frozen-feature subset | `src/semrember.cr`, `Memory#subset`, `#subset_by_labels` |
| candidates, H, encoding, plan | `src/shards.cr` |
| sequential at-most budget | `src/shards.cr`, `add_at_most_k` |
| NuWLS solve | `src/maxsat.cr` |
| CLI | `tools/shard_plan.cr` |
| tests | `spec/subset_spec.cr`, `spec/shards_spec.cr` |
