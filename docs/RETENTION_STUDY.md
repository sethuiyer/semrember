# Pilot study: does decision power define the hot working set?

[Influence derivation](INFLUENCE.md) · [Engine design](DESIGN.md)

## Question

If a semantic shard must keep only \(K\) live memories, should it retain them by
decision power rather than recency, frequency, or leverage?

This is the first small study. It is a pilot, not a product claim.

## Protocol

Corpus: the Pokémon name → primary-type CSV (697 rows, 18 labels).

1. Shuffle with a fixed seed.
2. Hold out 20% (139 names) for evaluation.
3. Train the full model on the remaining 558 names.
4. Build a query corpus of 100 perturbed training names (random word deletion or
   a `mut####` replacement).
5. Compute leave-one-out influence over that query corpus.
6. Rank every training memory by six strategies:
   - **recency** — later rows are retained.
   - **random** — fixed seed.
   - **frequency** — number of query-corpus queries for which the memory is the
     nearest stored key.
   - **leverage** — ridge leverage \(h_i=1-\lambda A_{ii}\).
   - **Banzhaf** — mean leave-one-out score influence over the query corpus.
   - **hybrid** — \(0.7\cdot\) normalized Banzhaf \(+0.3\cdot\) normalized recency.
7. For each \(K\), fit a fresh model on the selected \(K\) memories.
8. Measure agreement with the complete model on the held-out 139 names:
   - **decision agreement** — same top-1 label.
   - **escalation agreement** — same novelty-threshold decision at 0.55.

## Results

Decision agreement:

| K | random | frequency | leverage | Banzhaf | hybrid |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.129 | 0.115 | 0.122 | 0.115 | 0.101 |
| 20 | 0.173 | **0.223** | 0.194 | 0.173 | 0.137 |
| 40 | 0.223 | **0.288** | 0.201 | 0.144 | 0.173 |
| 80 | 0.273 | **0.324** | 0.252 | 0.281 | 0.266 |
| 120 | 0.338 | 0.331 | 0.302 | **0.374** | 0.288 |
| 200 | 0.439 | **0.446** | 0.374 | 0.417 | 0.374 |
| 300 | 0.547 | 0.532 | 0.511 | **0.576** | 0.532 |
| 400 | 0.662 | 0.691 | 0.604 | **0.705** | 0.676 |
| 500 | 0.813 | 0.835 | 0.863 | **0.885** | 0.806 |
| 558 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Escalation agreement:

| K | random | frequency | leverage | Banzhaf | hybrid |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.849 | 0.849 | 0.849 | 0.849 | 0.849 |
| 40 | 0.856 | 0.863 | 0.849 | 0.856 | 0.856 |
| 120 | 0.863 | 0.871 | 0.863 | **0.878** | 0.856 |
| 200 | 0.863 | **0.892** | **0.892** | 0.878 | 0.849 |
| 300 | 0.863 | 0.942 | 0.899 | **0.942** | 0.878 |
| 400 | 0.906 | 0.957 | 0.914 | **0.957** | 0.928 |
| 500 | 0.978 | 0.986 | 0.964 | **0.986** | 0.978 |

Influence pipeline for this run: 100 queries, \(N=558\), total ~60 ms,
of which the dense \(AK\) product is ~50 ms.

## Findings

### Banzhaf is useful, but not universally best

The hypothesis “retain by decision power” is **not confirmed** as a universal
rule.

- At very small \(K\) (10–40), **frequency** wins. The shard is so small that
  covering common query patterns matters more than pivoting near the current
  decision boundary.
- At middle and larger \(K\) (120–500), **Banzhaf** wins on decision agreement.
- The **hybrid** is never the best in this run. The simple 70/30 blend is not
  the right combination; it needs a better normalization or a learned weight.
- Leverage is weak as a retention score. High-leverage memories are not
  necessarily pivotal for the query distribution.

### Escalation agreement is relatively flat

Escalation agreement stays around 0.85–0.99. The novelty gate is mostly driven
by the lexical feature space, not by which individual memories survive. This is
consistent with the design: novelty is a property of the feature manifold, not
a vote count.

### A surprising coupling result

In the Pokémon influence run, `pikachu` was pivotal for some `mooshikachu`
variants even though `mooshikachu` was inserted as a new memory. Memories are
not additive players; the dual inverse couples them. This is why “similar to
many queries” and “high norm” are not substitutes for decision power.

## Limitations

- The Pokémon corpus is a weak testbed. Names are arbitrary strings; held-out
  top-1 is already low, and the query distribution is synthetic.
- The study uses one 80/20 split and one query-corpus seed.
- The hybrid score is a simple hand-tuned blend.
- Escalation agreement uses a fixed 0.55 novelty threshold, not a calibrated
  operating point.
- This is a pilot. The next study should use de-identified incident notes or
  support tickets with a real query distribution and multiple splits.

## Practical recommendation

Do not replace frequency with Banzhaf blindly.

A defensible default for a bounded shard:

1. Keep a **frequency/coverage floor** for small \(K\).
2. Rank the remainder by **Banzhaf influence**.
3. Pin high-risk memories (injury, fraud, complaint, total loss) regardless of
   score.
4. Recompute influence periodically, not on every write.
5. Measure on the customer's query distribution before choosing a blend.

## Reproduce

```bash
make retention POKEMON_CSV=~/semrember/build/pokemon.csv
make pokemon-banzhaf POKEMON_CSV=~/semrember/build/pokemon.csv
make banzhaf-mc
make bench-banzhaf BANZHAF_N=1000
```
