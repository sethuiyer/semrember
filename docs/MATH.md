# Core mathematics of semrember

[README](../README.md) · [Engine design](DESIGN.md) · [Influence](INFLUENCE.md) · [AMEM format](../FORMAT.md)

This is the mathematical model implemented by `src/semrember.cr` and
`src/influence.cr`, with the numerical caveats that matter in practice.

Notation: `N` observations, `D = D_tfidf + D_static` features, `T` labels,
`Q` queries, ridge penalty `λ` (default `0.05`).

| Symbol | Meaning |
|---|---|
| `x_i ∈ R^D` | feature vector of training observation `i` |
| `y_i ∈ {0,1}^T` | one-hot label |
| `X ∈ R^(N×D)`, `Y ∈ R^(N×T)` | stacked rows |
| `W ∈ R^(T×D)` | prediction weights |
| `K = XXᵀ + λI_N` | dual Gram |
| `A = K⁻¹` | dual inverse |
| `B = AY ∈ R^(N×T)` | label-space dual |
| `k(q) = Xx(q)` | query–train kernel |
| `s(q) = Wx(q) = Bᵀk(q)` | label scores |

---

## 1. Feature map

### Lexical half

Lowercase ASCII bytes only, then take every byte n-gram of length `1..5`.
Let `V` be the vocabulary of n-grams with document frequency `>= min_df`,
sorted by byte value. For a term `g ∈ V`:

```
tf_g   = 1 + ln(c_g)
idf_g  = ln((1 + N) / (1 + df(g))) + 1
u_g    = tf_g * idf_g
```

where `c_g` is the count in the text and `df(g)` is the number of observations
containing `g`. Then

```
ũ = u / (||u||_2 + 1e-9)
```

This is sublinear TF, smoothed IDF, and an L2-normalized lexical vector.

### Static half

An optional encoder returns `z ∈ R^D_static`, mean-pooled and L2-normalized.
The joint vector is

```
j = sqrt(||ũ||_2² + ||z||_2²) + 1e-9
x = [ ũ / j ; z / j ]
```

So `||x|| ≈ 1`. Without an encoder, only the lexical half exists. The
vocabulary and IDF are fixed at fit time; live writes use only existing
coordinates.

Byte n-grams can split UTF-8 codepoints. That is intentional and is part of the
typo-tolerance story.

---

## 2. Ridge model

One-vs-all ridge regression:

```
min_W  Σ_i ||y_i - Wx_i||_2² + λ ||W||_F²
```

Normal equations:

```
W (XᵀX + λI) = YᵀX
  ⇒ W = YᵀX (XᵀX + λI)⁻¹
```

The dual form avoids inverting a `D×D` matrix:

```
K = XXᵀ + λI
A = K⁻¹
W = Yᵀ A X
```

Prediction:

```
s(q) = Wx(q) = Yᵀ A Xx(q) = Bᵀ k(q)
B = AY
```

Classification is `argmax_c s_c(q)`, ties to the lowest label index. Scores are
raw regression outputs, not probabilities.

### Solvers

| Solver | State | Update |
|---|---|---|
| `dual` (default) | `A`, `B`, `W` | exact ridge; supports hard set/delete |
| `rls` | `P = (XᵀX + λI)⁻¹` | recursive least squares |
| `delta` | `W` | 30 sweeps of `W_c ← W_c + β(1[c=y] - W_c x)xᵀ`, `β = 0.3` |

RLS:

```
g = Px / (1 + xᵀPx)
W ← W + (y - Wx) gᵀ
P ← P - g (Px)ᵀ
```

In exact arithmetic RLS is ridge. The implementation is Float32 and skips
observations with denominator `<= 1e-12`. `delta` is not ridge at all; it is a
fixed-step delta rule.

Only `dual` is writable and supports the `A`-based metrics below. The current
`novelty`/`influence` paths call `@a.not_nil!`; for `rls`/`delta` that raises
`NilAssertionError` instead of a domain error.

---

## 3. Exact dual updates

Let `P_i = (AX)_{i,:} ∈ R^D`, so `P_i x(q) = z_i(q)`, and let `B_i` be row `i`
of `B`.

### Insert observation `(x, y)`

```
k = Xx
a = Ak
σ = λ + ||x||_2² - kᵀa
```

```
A' = [ A + aaᵀ/σ    -a/σ   ]
     [   -aᵀ/σ       1/σ   ]
```

The prediction update is a rank-one RLS step:

```
r  = (x - Xᵀa) / σ
W' = W + (y - Wx) rᵀ
```

Here `r = Px / (1 + xᵀPx)` is the RLS gain, written in dual form.

### Relabel observation `i` from `c` to `c'`

`A` is unchanged. Only two rows of `W` move:

```
W'_{c'} = W_{c'} + P_i
W'_{c}  = W_{c}  - P_i
```

If `c = c'`, it is a no-op. If a key has multiple observations, apply to each,
then prune unused label rows.

### Delete observation `i`

```
A' = A_RR - A_Ri A_iR / A_ii
c_i = B_i / A_ii
```

```
W' = W - c_i P_iᵀ
```

Equivalently, the per-query score change is

```
Δ_i(q) = s(q; M \ i) - s(q; M) = -z_i(q) c_i
```

This is the central identity behind influence, poison checking, and compaction.
It is exact in exact arithmetic; the code stores `W` in Float32, so repeated
edits accumulate rounding.

`delete_many` deletes rows in descending index order, downdating `A` and the
novelty inverse each time, and prunes labels at the end. The influence oracle
calls `delete_one` and does not prune labels, which is a subtle semantic
difference from the served model.

---

## 4. Readouts

### Similarity

Both rows and queries are (approximately) unit norm, so

```
sim(q, i) = x_iᵀ x(q) = cos(x_i, x(q))
```

The server deduplicates by key and joins hits back to the exact record store.

### Novelty (lexical leverage)

Novelty is measured on the lexical half only:

```
q_L = ũ / ||ũ||_2
k_L = X_L q_L
G_L = X_L X_Lᵀ + λI
```

```
ν(q) = 1 - k_Lᵀ G_L⁻¹ k_L  ∈ [0,1]
```

Why `[0,1]`: `X_Lᵀ(X_LX_Lᵀ + λI)⁻¹X_L ≺ I`, so the quadratic form is at most
`||q_L||² = 1`.

If there is no static encoder, `X_L = X` and `G_L = K`, so `G_L⁻¹ = A` is
reused. If there is a static encoder, `G_L` is a separate lexical Gram; the
cached inverse is maintained by the same insert/delete formulas.

Subtlety: with a static encoder, the rows' lexical parts are scaled by the
joint norm `j`, while the query is renormalized to unit lexical norm. So novelty
is not identical to a purely lexical model's novelty; adding an encoder shifts
the scale.

### Ridge leverage

```
h_i = 1 - λ A_ii
```

The standard identity

```
A_ii = (1 - h_i) / λ
```

is why `A_ii` is the leave-one-out denominator.

---

## 5. Decision gates

For a query with at least two candidate labels:

```
m(q) = (s_(1) - s_(2)) / (|s_(1)| + |s_(2)|)  ∈ [0,1]
```

If there is exactly one candidate, `m = 1`; if none, `m = 0`.

With thresholds `τ_m` (`min_margin`, default `0.35`) and `τ_ν`
(`min_novelty`, default `0.55`):

```
handled ⇔ m(q) >= τ_m  and  ν(q) <= τ_ν  and  candidates exist
```

Reason precedence:

| Condition | `reason` |
|---|---|
| `m < τ_m` and `ν > τ_ν` | `both` |
| `m < τ_m` only | `ambiguous` |
| `ν > τ_ν` only | `out_of_domain` |
| otherwise | `handled` |

The gates measure different things: a scale-free score gap and a lexical
leverage. Neither is a calibrated probability.

---

## 6. Influence

Define the empirical leave-one-out pivotal rate over a query corpus `Q`:

```
β_i = (1/Q) Σ_q 1[ argmax_c s_c(q) != argmax_c s_c^(-i)(q) ]
```

with

```
s^(-i)(q) = s(q) - z_i(q) c_i
z_i(q)    = (Ak(q))_i
c_i       = B_i / A_ii
```

Continuous influence:

```
I_i = (1/Q) Σ_q ||z_i(q) c_i||
    = ||c_i|| · (1/Q) Σ_q |z_i(q)|
```

This is not the classical coalition Banzhaf index; it is the marginal effect of
deleting one observation from the full model, averaged over queries.

### Safe pruning bound

For the raw winner–runner-up margin `μ(q) = s_(1) - s_(2)` and
`span(c_i) = max_c c_i,c - min_c c_i,c`:

```
s'_c - s'_w = (s_c - s_w) + z_i(q)(c_i,w - c_i,c)
```

so

```
|z_i(q)| span(c_i) < μ(q) - ε
  ⇒ memory i cannot change the top-1 for q
```

The implementation uses this to skip the expensive `T`-vector update. Rows with
`A_ii <= 1e-12 max_j A_jj` are marked unstable and fall back to exact deletion.

Computational shape: `B = AY` is `O(N² + NT)`, `Z = A K_q` is `O(N²Q)`,
evaluation is `O(NQ)` pruned or `O(NQT)` worst case. Memory is
`O(N² + NQ + NT)`.

Numerical caveat: influence uses Float64 `A`-derived scores; the served `query`
uses Float32 `W`. Near-ties can disagree.

---

## 7. Compaction

Let `K` be the candidate keys whose every observation has `β_i <= τ`, excluding:

- pinned keys,
- keys covering at least `coverage_floor` queries as the nearest memory,
- keys needed to keep `keep_per_label` per label (chosen by largest mean
  influence).

For a batch `B ⊆ K`, verification is

```
D(q; M \ B) = D(q; M)   for all q ∈ Q
```

The algorithm tries the batch on a copy:

- if all decisions are preserved, delete it from the live model;
- otherwise split in half and recurse;
- a singleton that fails is skipped.

With `verify: true`, every accepted batch preserves decisions, so the final
`changed == 0` / `verified` flag is structurally true. It is a consistency check
over the same corpus, not independent held-out validation. With `verify: false`,
the model is mutated first, then the flag is computed; a 409 can therefore
follow a destructive change, with no rollback.

`β = 0` is a leave-one-out property, not a group property. Recursive
verification exists because deleting a set can change decisions even when every
member has `β = 0`.

---

## 8. Poison check

For a proposed write, build `M'` by applying `set` to a copy. Let

```
P = { q ∈ Q : argmax s(q; M) != argmax s(q; M') }
```

```
S = (1/|P|) Σ_{q∈P} ( 1 - cos(x_new, x_q) )  ∈ [0,1]
```

Verdict:

```
clean       if |P| = 0
suspicious  if S >= 0.5 and (ν >= 0.8 or |{q∈P: cos < 0.3}| >= 2)
uncertain   if S >= 0.3
clean       otherwise
```

This is why the canonical exact-key overwrite is rated clean: if the written key
equals the query, cosine `= 1`, so `S = 0`, even though the top-1 flipped. It is
a top-1 flip detector with a similarity penalty, not an access-control boundary.

Cumulative drift is a heuristic: over the last 16 checks, count `suspicious`;
`>= 3` rejects subsequent `reject_suspicious` writes globally. That is stateful
telemetry, not a mathematical guarantee.

---

## 9. What is exact, and what is not

| Quantity | Numeric reality |
|---|---|
| `A` | Float64, Cholesky inverse; Gauss-Jordan fallback |
| `W` during fit | Float64 accumulation, stored Float32 |
| live `W` | Float32; every mutation rounds |
| AMEM `W` | IEEE binary16 (round-to-nearest-even) |
| scores | Float32 dot product |
| novelty | Float64 `A`, Float64 Gram |
| influence fast path | Float64; exact-deletion fallback |
| `β_i` | empirical over finite `Q`, not a distribution |
| poison verdict | heuristic over a caller-influenced query buffer |
| compact `verified` | exact on the supplied corpus; structurally true with `verify: true` |

Because `x_i` is normalized, `K_ii ≈ 1 + λ`, so
`A_ii >= 1/(1 + λ) ≈ 0.95` at the default `λ = 0.05`. The `1e-12` “unsafe”
guards in insert/delete are effectively unreachable in normal operation;
genuine drift in `A` is not detected outside the test-only `writer_residual`.
Conversely, a legal tiny `λ` can pass the guards and amplify roundoff by
`1/λ`.

### AMEM quantization

The portable file stores

```
Ŵ_{c,d} = f16(W_{c,d})
```

row-major, plus vocabulary, IDF, labels, and a Bloom filter. Prediction from a
loaded model is `s(q) = Ŵx(q)`. Float16 has about 11 bits of significand, so
saved scores differ slightly from live Float32 scores. The file intentionally
stores no observations, keys, TTLs, or writer state; a loaded model is a reader.

---

## 10. Appendix: durable-store math (not model math)

Key separation and AEAD:

```
k_store = BLAKE2b-KDF(k, id = 1, context = "SemrStor")
C       = XChaCha20-Poly1305_{k_store}(P, nonce_24, AD = "SMRENC01")
```

The plaintext payload is

```json
{"version":3,"counter":N,"encoder":...,"records":{...},"expiry":{...}}
```

with a monotonic counter authenticated inside the ciphertext. The snapshot is
committed via temp file, fsync, atomic rename, and directory fsync. The counter
is a rollback floor, not a model quantity.

---

## Source map

| Math | Code |
|---|---|
| feature map, TF-IDF, normalization | `src/semrember.cr:375-404` |
| ridge objective and solvers | `src/semrember.cr:464-591` |
| dual state and exact set/delete | `src/semrember.cr:741-1020` |
| novelty and leverage | `src/semrember.cr:680-730`; `src/influence.cr:505-507` |
| influence and safe bound | `src/influence.cr:102-378` |
| decision gates | `src/server.cr:470-521` |
| AMEM quantization | `src/semrember.cr:1027-1127`; `FORMAT.md` |
| encrypted snapshot | `src/encrypted_store.cr`; `docs/DURABILITY.md` |
