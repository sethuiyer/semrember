# 15-row worked example: counterfactual deletion by rank-one algebra

[Influence derivation](INFLUENCE.md) · [Retention study](RETENTION_STUDY.md)

This is a complete numeric example on 15 memories. It shows the exact
leave-one-out identity and verifies it against actual deletion.

## Corpus

Fit settings: `solver=:dual`, `min_df=1`, `rls_reg=0.05`.

| i | key | label |
|---:|---|---|
| 0 | `disk full` | free_disk |
| 1 | `no space left` | free_disk |
| 2 | `filesystem full` | free_disk |
| 3 | `host down` | fix_dns |
| 4 | `dns timeout` | fix_dns |
| 5 | `cannot resolve host` | fix_dns |
| 6 | `cert expired` | renew_cert |
| 7 | `tls handshake failed` | renew_cert |
| 8 | `x509 expired` | renew_cert |
| 9 | `cache stale` | clear_cache |
| 10 | `stale cache entry` | clear_cache |
| 11 | `cache corruption` | clear_cache |
| 12 | `disk full on host` | free_disk |
| 13 | `dns cache stale` | fix_dns |
| 14 | `cert cache expired` | renew_cert |

Result: \(N=15\), \(D=491\), \(T=4\).

Label order:

```text
0 = free_disk
1 = fix_dns
2 = renew_cert
3 = clear_cache
```

## Query

```text
q = "cache stale"
```

Current Float64 scores from \(s(q)=Wq\):

```text
s(q) = [0.005173103745621,
        0.071304985086933,
        0.001688537036118,
        0.905285702295425]
```

Winner: `clear_cache` (label 3).

## Delete memory 9: `"cache stale" -> clear_cache`

The relevant quantities are:

```text
A_ii = 2.358640090330283
z_i(q) = (A X q)_i = 0.882067995483487
```

The label vector \(B=AY\) for row 9:

```text
B_i = [-0.103462074912420,
       -1.426099701738668,
       -0.033770740722354,
        1.894285954091535]
```

The rank-one direction:

```text
c_i = B_i / A_ii
    = [-0.043865138787636,
       -0.604627941153569,
       -0.014317886336624,
        0.803126327690918]
```

The counterfactual score change:

\[
\Delta_i(q)=-\frac{z_i(q)}{A_{ii}}B_i
\]

```text
delta = -z_i * c_i
      = [ 0.038692035042015,
          0.533322956066637,
          0.012629349300507,
         -0.708412029986342]
```

Fast counterfactual scores:

```text
s(q; M\9) = s(q) + delta
          = [0.043865138787636,
             0.604627941153569,
             0.014317886336624,
             0.196873672309083]
```

Winner after deletion: `fix_dns` (label 1).

So memory 9 is **pivotal** for this query: deleting it flips
`clear_cache -> fix_dns`.

## Exact deletion check

Actual deletion of observation 9 from a copy gives:

```text
exact = [0.043865138787636,
         0.604627941153570,
         0.014317886336624,
         0.196873672309082]
```

Maximum absolute error:

```text
4.16e-16
```

The fast identity and the exact delete path agree to machine precision.

## Banzhaf over the 15-row corpus

Query corpus:

```text
["disk full", "cert expired", "cache stale", "dns timeout",
 "disk full on host", "cert cache expired", "unknown"]
```

Only memory 9 is pivotal, on the `"cache stale"` query:

| memory | pivotal queries | beta | mean influence |
|---|---:|---:|---:|
| `cache stale` (9) | 1 | 0.142857 | 0.137730 |
| all other 14 memories | 0 | 0.000000 | 0.0025–0.0963 |

The other memories still have nonzero continuous influence — they move the
scores slightly — but none changes the top-1 decision on this query set.

## Why this example matters

- `A_ii` is not a heuristic weight. It is the exact leave-one-out denominator.
- \(z_i(q)\) is a scalar query influence, computed from the existing inverse.
- \(B_i/A_{ii}\) is the precomputed label direction.
- The full vector update is rank one.
- The top-1 decision can be evaluated without constructing \(A^{(-i)}\) or
  \(W^{(-i)}\).
- The exact delete path is used only as a correctness oracle.

## Reproduce

The example is exercised by the influence test suite and can be reproduced with:

```crystal
mem = Semrember::Memory.new(solver: :dual, min_df: 1, rls_reg: 0.05).fit(data)
terms = mem.influence_terms("cache stale", 9)
terms.fast_scores
terms.exact_scores
```

See `spec/influence_spec.cr` for the randomized and edge-case equivalents.
