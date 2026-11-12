# Decision power and leave-one-out influence

[README](../README.md) · [Engine design](DESIGN.md) · [Retention pilot study](RETENTION_STUDY.md) · [Compaction pilot](COMPACTION_STUDY.md) · [15-row worked example](WORKED_EXAMPLE.md)

This document defines the influence tooling, derives the exact leave-one-out
shortcut, and reports what the experiments actually showed.

## Two different quantities

**Leave-one-out pivotal power** (implemented as `Memory#banzhaf_influence`):

\[
\beta_i =
\Pr_{q\sim Q}\left[D(q;M)\neq D(q;M\setminus\{i\})\right].
\]

This measures influence from the **full fitted model**. It is not the classical
coalition Banzhaf index.

**Classical coalition Banzhaf** (implemented for small experiments in
`tools/banzhaf_mc.cr`): for a game \(v(S)\) over subsets of memories,

\[
\beta_i^{\mathrm{coal}} =
\frac{1}{2^{N-1}}\sum_{S\subseteq [N]\setminus\{i\}}
\left[v(S\cup\{i\})-v(S)\right].
\]

The tool names this separately. Do not use one number as if it were the other.

## Exact leave-one-out derivation

Semrember's dual writer maintains

\[
K=XX^\top+\lambda I,\qquad A=K^{-1},\qquad W=Y^\top AX,
\]

and predicts \(s(q)=Wq\).

Deleting observation \(i\) and retaining \(R=[N]\setminus\{i\}\) gives the
principal-submatrix inverse

\[
A^{(-i)}=A_{RR}-\frac{A_{Ri}A_{iR}}{A_{ii}}.
\]

Let \(x_i\) be row \(i\) of \(X\), \(y_i\) row \(i\) of \(Y\), and

\[
P_i=(AX)_{i,:},\qquad k(q)=Xq,\qquad z_i(q)=(Ak(q))_i.
\]

The deletion identities already used by the writer are

\[
P^{(-i)}=P_R-\frac{A_{Ri}}{A_{ii}}P_i,
\]

\[
W^{(-i)}=W-\left(y_i^\top+Y_R^\top t\right)P_i,
\qquad t=\frac{A_{Ri}}{A_{ii}}.
\]

Define \(B=AY\). Since

\[
y_i+\sum_{j\ne i}\frac{A_{ij}}{A_{ii}}y_j
=\frac{B_i}{A_{ii}},
\]

the change in the weight matrix is rank one:

\[
W^{(-i)}-W
=-\frac{B_i}{A_{ii}}\otimes P_i.
\]

Therefore the change in the score vector is

\[
\boxed{
\Delta_i(q)
=s(q;M\setminus\{i\})-s(q;M)
=-\frac{z_i(q)}{A_{ii}}B_i
}
\]

where \(B_i=(AY)_i\) is a \(T\)-vector and \(z_i(q)\) is a scalar.

This is the central shortcut. No \(A^{(-i)}\), \(W^{(-i)}\), or new `Memory`
object is needed per memory.

The label influence magnitude is

\[
\|\Delta_i(q)\| = |z_i(q)|\cdot \|B_i/A_{ii}\|.
\]

The mean continuous influence is

\[
I_i=\mathbb E_{q\sim Q}\|s(q;M)-s(q;M\setminus i)\|
=\|B_i/A_{ii}\|\cdot \mathbb E_{q\sim Q}|z_i(q)|.
\]

## Connection to classical ridge leave-one-out

Let \(h_{ii}=x_i^\top(X^\top X+\lambda I)^{-1}x_i\) be the ridge leverage.
The identity

\[
A_{ii}=\frac{1-h_{ii}}{\lambda}
\]

makes the denominator \(A_{ii}\) the usual leave-one-out scaling. The vector
\(B_i/A_{ii}\) is the label-residual direction for observation \(i\), and
\(z_i(q)=A_{i,:}Xq\) is the corresponding query influence. This is the exact
ridge PRESS/leave-one-out structure, not a first-order influence
approximation.

## Algorithm

1. Encode the query corpus once.
2. Build \(K_{:,q}=Xq\) for every query — sparse feature dot products.
3. Precompute \(B=AY\) in \(O(N^2+NT)\).
4. Precompute \(A_{ii}\), \(c_i=B_i/A_{ii}\), \(\|c_i\|\), and
   \(\mathrm{span}(c_i)=\max c_i-\min c_i\).
5. Compute \(Z=AK\) once. This is the only dense \(O(N^2Q)\) operation.
6. For each query, find the current winner and runner-up margin.
7. For each memory, evaluate the scalar \(z_i(q)\).
8. If \(|z_i(q)|\cdot\mathrm{span}(c_i)<\text{margin}\), the memory cannot
   change the top-1 decision; skip the full \(T\)-vector update.
9. Otherwise form \(s' = s - z_i(q)c_i\), recompute the winner, and count a
   pivotal query.
10. Accumulate mean/max influence and example queries.

### Safe pruning bound

For the current winner \(w\), runner-up margin \(m\), and any other label \(c\),

\[
s'_c-s'_w
=(s_c-s_w)+z_i(q)(c_i[w]-c_i[c])
\le -m+|z_i(q)|\,\mathrm{span}(c_i).
\]

So if

\[
|z_i(q)|\,\mathrm{span}(c_i)<m,
\]

no label can overtake the winner and memory \(i\) is not pivotal for that
query. The implementation uses a small epsilon to avoid tie/boundary flips.

## Complexity

\(N\) memories, \(T\) labels, \(Q\) queries:

| Phase | Cost |
|---|---|
| Encode queries | \(O(Q\cdot\text{encode})\) |
| Query/train kernel \(K\) | \(O(Q\cdot \mathrm{nnz}(X))\) |
| \(B=AY\) | \(O(N^2+NT)\) |
| \(Z=AK\) | \(O(N^2Q)\) |
| Evaluation | \(O(NQ)\) pruned, \(O(NQT)\) worst case |
| Memory | \(O(N^2+NQ+NT)\) |

The naive exact-delete-and-query baseline is \(O(QN)\) model mutations, each
roughly \(O(N^2)\) for deletion plus query cost: effectively \(O(QN^3)\).
The shortcut removes the \(N\) factor from the mutation dimension.

## Numerical stability

The expression divides by \(A_{ii}\). The code computes a relative floor
`1e-12 * max(A_ii)`. Memories below that floor are marked `unstable=true` and
fall back to the exact deletion oracle for their query evaluations. The
fallback is slower but never silently returns the unreliable rank-one value.

The test suite compares the fast formula against an independent exact deletion
on:

- randomized memories and queries
- weighted duplicate observations
- nearly identical features with different labels
- insertion, deletion, and label overwrite
- single-label datasets
- static embeddings
- empty query sets

The score vectors agree to about `1e-10` or better.

## Empirical and Monte Carlo modes

`Memory#banzhaf_influence(queries)` is the empirical leave-one-out mode. It
returns key, \(\beta_i\), pivotal query count, mean and max influence,
examples, \(A_{ii}\), and an instability flag.

`tools/banzhaf_mc.cr` is the classical coalition mode. It samples coalitions
(and caches subset decisions) and reports an estimate plus a 95% confidence
interval. It adaptively stops per memory when the interval is narrow enough.
It is practical only for small memory sets because each coalition requires an
exact subset model.

## Pokémon demonstration

After `set("mooshikachu", "Normal")` on the Pokémon name → primary type cache:

```text
memory           beta     pivotal  mean-I
mooshikachu      0.556    5        0.523
pikachu          0.333    3        0.225
raichu           0.000    0        0.110
pichu            0.000    0        0.089
smoochum         0.111    1        0.135
```

`mooshikachu` is pivotal for its own spelling basin and not for
`pikachu`/`raichu`. The counterfactual scores match exact deletion to
`~1e-15`.

Surprise: `pikachu` also has pivotal power for some mooshikachu variants. The
memories are not additive players; the dual inverse couples them. This is why
“similar to many queries” is not the same as decision power.

## Benchmarks

With `-Dlapack`, Q=200 synthetic queries:

| N | fit | influence total | kernel | matmul | eval | naive est. | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1.1 s | 0.42 s | 0.28 s | 0.11 s | 0.9 ms | 18.2 s (direct) | 43.8× |
| 2,000 | 5.5 s | 1.03 s | 0.57 s | 0.43 s | 1.8 ms | 61.2 s (direct) | 59.7× |
| 4,000 | 35 s | 3.00 s | 1.18 s | 1.74 s | 4.5 ms | 295 s (direct) | 98.3× |
| 6,000 | 102 s | 5.77 s | 1.64 s | 3.99 s | 7.6 ms | 1,119 s (estimated) | 194× |

The 1k, 2k, and 4k rows use a direct full naive delete-and-query run. The 6k
row extrapolates from a ten-memory sample because a full naive run takes about
19 minutes. The speedup grows with N because the shortcut removes the
per-memory mutation.

## Retention experiment

`tools/retention_experiment.cr` trains on 80% of the Pokémon corpus, computes
influence on 100 perturbed training names, and compares retention strategies on
the held-out 20%.

Selected decision-agreement results:

| K | random | frequency | leverage | Banzhaf | hybrid |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.129 | 0.115 | 0.122 | 0.115 | 0.101 |
| 40 | 0.223 | **0.288** | 0.201 | 0.144 | 0.173 |
| 120 | 0.338 | 0.331 | 0.302 | **0.374** | 0.288 |
| 200 | 0.439 | **0.446** | 0.374 | 0.417 | 0.374 |
| 300 | 0.547 | 0.532 | 0.511 | **0.576** | 0.532 |
| 400 | 0.662 | 0.691 | 0.604 | **0.705** | 0.676 |
| 500 | 0.813 | 0.835 | 0.863 | **0.885** | 0.806 |

The hypothesis “retain by decision power” is **not universally true** here.
Frequency wins at very small K; Banzhaf wins at middle and large K; hybrid is
not the best strategy. The Pokémon corpus is also a weak testbed because names
are arbitrary strings and held-out top-1 is already low. The honest conclusion
is that Banzhaf is a useful signal, not a universal replacement for frequency
or recency.

## Classical mathematics vs new engineering

Classical:

- ridge leave-one-out / PRESS residuals
- hat matrix and leverage
- Sherman–Morrison and principal-submatrix inverse identities
- Banzhaf power and Monte Carlo confidence intervals

Semrember-specific engineering:

- applying the exact leave-one-out formula at the **memory** level
- using the existing dual inverse and sparse feature store as the precomputed
  state
- query-distribution pivotal power rather than a single query
- margin-based pruning with a safe bound
- hot-working-set retention as a use for the influence score
- separating empirical leave-one-out power from classical coalition Banzhaf

## Poison-check limits

The poison check is a diagnostic, not an access-control boundary.

- An attacker who writes a key lexically similar to the target queries can
  lower the dissimilarity signal.
- A single clean write does not prove absence of cumulative drift.
- The server keeps a short history of recent poison checks and treats three
  `suspicious` verdicts in the recent window as a **heuristic cumulative drift
  signal**, not a detection guarantee.
- It does not replace provenance, rate limits, review, or a trust boundary on
  writers.
- It measures model sensitivity, not malicious intent.
- The server's rolling query buffer is caller-influenced. An unauthenticated
  caller can submit queries first and shape the detector's reference
  distribution. Treat that buffer as telemetry, not trusted policy state.
- For a new key the counterfactual is absence -> insertion; for an existing key
  it is old label -> proposed label.

## HTTP surface

The live server exposes the governance layer:

- `POST /v1/influence` — leave-one-out pivotal power for a query list
- `POST /v1/compact` — verified recursive compaction
- `PUT /v1/kv/{key}` with `"poison_check": true` — per-write poison check
- `PUT /v1/kv/{key}` with `"reject_suspicious": true` — reject a suspicious
  write with HTTP 409

The poison check uses the last 128 SIMILAR / ESCALATE / NOVELTY inputs as its
query corpus. See [API.md](../API.md) for the exact JSON shapes.

## Reproduce

```bash
# exact leave-one-out influence
crystal run --release examples/pokemon_banzhaf.cr -- ~/semrember/build/pokemon.csv

# classical coalition Banzhaf + Monte Carlo CI
crystal run --release tools/banzhaf_mc.cr

# influence benchmark
crystal run --release -Dlapack tools/bench_banzhaf.cr -- 1000 2000 4000 6000

# retention curves
crystal run --release tools/retention_experiment.cr -- ~/semrember/build/pokemon.csv
```
