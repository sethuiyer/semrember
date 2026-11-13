# Standalone weight index (PR A)

This change adds an opt-in building block, `Semrember::Mips::Index`, for
best-bound-first search over Float32 weight rows. It does **not** change
`Memory#query`, `scores`, the server, AMEM, influence, novelty, similarity,
or the ridge writer. SUTRA/shard selection and router training are separate
work. Nothing is enabled automatically.

## Contract and use

```crystal
require "./src/weight_index"

config = Semrember::Mips::Config.new(rank: 32, leaf_size: 16)
index = Semrember::Mips::Index.new(weights, label_count, dimension,
  config, version: model_version)
result = index.search(dense_features, 3, expected_version: model_version) do |ids|
  # Score the SAME immutable W snapshot. Preserve the original Float32
  # multiply/add sequence and feature order for each requested label.
  score_labels(ids)
end
if result.fallback?
  # Run the complete reference scorer. Partial hits are deliberately discarded.
else
  # Hits contain original Float32 scores, descending score / ascending label id.
  result.hits
end
```

The callback is a trusted part of this API: it must use the indexed weight
snapshot, matching Float32 feature values, at most `terms` multiply/add pairs,
and the reference arithmetic/order. `terms` defaults to D, conservatively
covering sparse lexical plus dense static scoring; larger values are permitted.
Duplicate feature coordinates must be consolidated consistently before using
this interface. A changed model must use a new version and a matching snapshot.

`topk == 0` and empty labels return empty results. Full-k, zero-dimensional or
zero queries, stale versions, unsupported floating-point environments,
nonfinite inputs/bounds/scores, possible Float32 intermediate overflow, and
exhausted work budgets request a full scan. Invalid shapes/configuration raise
`ArgumentError`; failed index construction must leave the caller on its full
scorer. Rank zero and full rank are supported for testing but may prune poorly.

Finite input weights are inspected at construction. Index geometry is owned
by the index; changing input weights or a supplied basis does not change it.
The index does not own the scoring callback or prevent the caller from mutating
its weight matrix. Results are exact only under the snapshot/callback contract.

## A bound for stored numbers, not idealized orthogonality

For any stored basis V and stored row coordinates h_c, define the *real*
residual e_c = w_c - V h_c. We do not assume h_c is an exact projection, nor
that V is exactly orthonormal. For u = V^T x:

    w_c^T x = h_c^T u + e_c^T x.

A node holds a stored center a, an upper radius rho >= max ||h_c-a||,
an upper residual E >= max ||e_c|| and an upper weight norm W >= max ||w_c||.
Consequently:

    w_c^T x <= a^T u + rho ||u|| + E ||x||.

The implementation encloses residual construction, radius/norm calculations,
query projection, and node-bound evaluation using Float64 interval arithmetic.
Each basic operation is widened to adjacent representable values, including
square root. Stored h and centers are treated as exact binary64 values; their
formation need not be exact because the residual and radius enclosures refer
to those stored values. There is no fixed `2^-40` construction allowance and
no configurable production epsilon.

Let n be the supplied operation-count upper bound and u32 = 2^-24. With
k = 2n and gamma = k*u32/(1-k*u32), use the conservative scorer allowance:

    gamma * W * ||x|| + k * 2^-149 / (1-k*u32).

The second term also accounts for propagation of absolute gradual-underflow
errors. Its denominator and the final arithmetic are rounded outward. Reject
nonpositive denominators. Before searching, an absolute-sum bound rules out
Float32 intermediate overflow, including overflow before later cancellation.

The final bound encloses the computed Float32 score. Prune only when it is
**strictly less** than the current kth actual Float32 score. Equality is
explored so lower-index ties cannot be lost. Finalists are scored exclusively
by the reference callback, and work-budget fallback returns no partial answer.

Supported execution requires IEEE binary32/binary64, correctly rounded basic
operations/square root, round-to-nearest, gradual underflow, and no fast-math
or reassociation. Linux rounding-mode and runtime underflow probes reject
unsupported environments. Index construction rejects unsupported environments; callers on other platforms must use their full scorer. The
floating-point mode must not change during construction or search. These
assumptions are part of the contract; fuzzing supports the implementation but
is not a universal proof about arbitrary compiler/platform behavior.

The tighter E*eta + D_v*||u|| variant is deliberately deferred.

## Basis and tree

A seeded randomized right subspace iteration and a small symmetric Jacobi
Rayleigh-Ritz solve choose the basis. Two-pass modified Gram-Schmidt removes
dependent columns. Numerical basis quality affects pruning efficiency, not
bound validity. Tests can supply arbitrary, nonorthonormal bases.

Nodes split on the maximum-variance projected coordinate and partition around
the median. Deterministic quickselect has a bounded partition budget and a
sort fallback: usual split work O(T log T), worst-case O(T log^2 T), plus
O(T r log T) vector work. Basis construction and outward residual calculation
are additional build costs. Best-bound-first traversal uses a max-heap with
node-id ties. Rank, leaf size and work budgets are explicit configuration.

`payload_bytes` estimates numeric arrays and node data only, not allocator,
object-header, temporary-build or reference-weight storage.

## Lifecycle: next integration change

This standalone PR records `Index#version` and rejects mismatched query
versions. It does not yet implement a Memory-owned version counter, a
publication lock, asynchronous rebuilds, CLI flags or `/v1/info` fields.

Integration must take W, labels and version under the same lock; build outside
the lock; then compare version and publish atomically under that lock. Queries
must hold a consistent model/index snapshot through exact callback scoring.
A version check followed by reading mutable W is insufficient. Writes,
pruning, adopt/load and copy operations invalidate the version; stale queries
use the reference scorer. No index is built on a request path or serialized
into AMEM. A race test belongs with that integration, not a simulated test of
an unimplemented publication mechanism.

## Verification and benchmarks

```
make mips-test
crystal spec --release spec/mips_spec.cr
make bench-mips MIPS_ARGS='--labels 1024 --dim 4096 --rank 32 --queries 50'
make bench-mips MIPS_ARGS='--labels 256 --dim 256 --mode gaussian'
```

The standalone suite checks every node against actual Float32 scores; random
sizes/ranks/scales; top-1/2/3 and bitwise finalist scores; exact-rational dot
error on subnormal/cancellation inputs; ties; arbitrary bases; empty cases;
mutation of construction arrays; stale versions; budgets; nonfinite values;
actual pruning; heap ordering and the added linear-algebra helpers.

The benchmark uses the reference feature-major Float32 accumulation order,
includes mandatory full scans after fallback, verifies parity outside timing,
and reports build time, numeric payload, visited nodes, scored labels and
fallback rate. It alternates execution order across repetitions and reports raw samples plus median times. The full baseline uses direct label iteration (no ids indirection) and also selects top-k. Modes include favorable
low-rank rows, Gaussian rows, near duplicates and sparse rows. These are
synthetic scoring measurements, not server or task-accuracy results.

Local measurements (Crystal 1.14.0, LLVM 19.1.6, --release, seed 0, top-3; ratio of median times from five alternating-order repetitions):

| Mode | T | D | Rank | Mean labels scored | Scoring speedup |
|---|---:|---:|---:|---:|---:|
| Low-rank | 1024 | 4096 | 32 | 16 | 11.81x |
| Gaussian | 256 | 256 | 16 | 256 | 0.70x |
| Near duplicates | 256 | 256 | 16 | 64 | 1.06x |
| Sparse | 256 | 256 | 16 | 256 | 0.43x |

All four had exact top-3 parity and zero budget fallbacks. Low-rank build time
was 17.84 seconds and numeric payload 1,368,792 bytes. Low-rank used 50 queries;
the other modes used 30. Raw samples are in [MIPS_BENCHMARK.json](MIPS_BENCHMARK.json).
These are host-load-sensitive observations on one synthetic seed. The random
and sparse cases are slower than the full scan. Earlier single-pass timings
varied substantially, which is why the harness reports samples and medians.

12x overall remains unestablished. Integration must measure the scoring
fraction f and realized scoring speedup S, then report
1 / ((1-f) + f/S), along with actual end-to-end latency. Random geometry can
make this index slower than a scan; keeping production behavior unchanged is
intentional.
