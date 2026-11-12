# Counterfactual compaction pilot

[Influence derivation](INFLUENCE.md) · [Retention study](RETENTION_STUDY.md) · [Worked example](WORKED_EXAMPLE.md)

## Goal

Remove memories whose leave-one-out pivotal power is zero on a query corpus,
without changing any decision on that corpus, then measure what happens on a
held-out set.

The tempting implementation is:

```text
1. beta = banzhaf_influence(queries)
2. delete every key with beta == 0
3. done
```

That is **not safe**. Zero beta is a leave-one-out property, not a group
property. Deleting many zero-beta memories together can change decisions
through interactions in the dual inverse.

The shipped `Memory#compact` therefore uses **recursive safe batches**:

```text
try to delete the whole candidate batch
  if verified -> accept
  else split the batch in half and recurse
  a single candidate is skipped if it changes any query decision
```

The verification model is the current model, and the query corpus is the
distribution being protected.

## Protocol

- 80/20 train/test split, fixed seed.
- Influence query corpus: 100 perturbed training texts.
- Compaction: `beta_threshold = 0.0`, `keep_per_label = 1`, `verify = true`.
- Coverage floor: keep any key that is the nearest memory for at least N query
  corpus items.
- Metric: decision agreement with the full model on the held-out 20%.
- Held-out data is **not** used by compaction; it is only used for evaluation.

## BBC News

Corpus: `BBC_News_Article_Train.csv`, 5 classes, capped at 1,200 rows.
Train \(N=960\), test 240.

| coverage floor | kept | pruned | verified | query changes | held-out agreement | time |
|---:|---:|---:|---|---:|---:|---:|
| 0 | 49 | 911 | yes | 0 | 89.6% | 45.6 s |
| 2 | 53 | 907 | yes | 0 | 91.7% | 49.6 s |

This is the strongest compaction result. It removes 95% of the training
memories while preserving every decision on the compaction corpus and 90–92%
of held-out decisions.

## Twitter airline sentiment

Corpus: `Tweets.csv`, 3 classes, capped at 1,000 rows.
Train \(N=800\), test 200.

| coverage floor | kept | pruned | verified | query changes | held-out agreement | time |
|---:|---:|---:|---|---:|---:|---:|
| 2 | 171 | 629 | yes | 0 | 60.0% | 6.7 s |

This is the important negative result. Compaction preserved every decision on
the perturbation corpus but dropped held-out agreement to 60%. The query
distribution was too narrow: it rewarded removing memories that were redundant
for perturbed training text but useful for held-out tweets.

## Interpretation

- Compaction can be extremely effective when the query corpus represents the
  deployment distribution.
- It can overfit the query corpus badly when the distribution is narrow.
- Zero-beta is not compositional; recursive verification is required.
- `compact` should be **scheduled or explicit**, never run on every `PUT`.
- A production policy should combine:
  - a coverage floor,
  - pinned high-risk keys,
  - held-out validation before accepting the compacted model,
  - a rollback path to the pre-compaction snapshot.

## Reproduce

```bash
crystal run --release tools/compact_experiment.cr -- \
  ~/Documents/Documents/Workspace/Datasets/BBC_News_Article_Train.csv \
  --text-col text --label-col category --max-rows 1200 --queries 100 --coverage-floor 2

crystal run --release tools/compact_experiment.cr -- \
  ~/Documents/Documents/Workspace/Datasets/Tweets.csv \
  --text-col text --label-col airline_sentiment --max-rows 1000 --queries 100 --coverage-floor 2
```
