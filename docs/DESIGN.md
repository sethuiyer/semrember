# Engine design and historical measurements

[Start with the developer quickstart](QUICKSTART.md).

## The writers

| solver | state | semantics |
|---|---|---|
| `delta` | `W` | iterative sweeps; the large-`D` fallback |
| `rls` | `P` (`D×D`) | exact ridge, recursive least squares |
| `dual` | `A` (`N×N`) | the same exact ridge solution in the kernel dual |

`dual` is the default because it is cheaper whenever `N < D` — the regime
this library targets — and because it supports **exact replacement**.

The Gram is built from sparse lexical features with a two-pointer dot product,
plus the dense static half when an encoder is present.  The inverse comes from
a Cholesky factorization, with Gauss-Jordan as a numerical fallback.  The
classifier is formed as `W = (A Y)ᵀ X`, so the `N×D` representer matrix is not
materialised.

Live state is `A` (N×N), `W` (T×D), and the sparse feature rows.  There is no
dense `N×D` observation cache.  Optional `-Dlapack` uses the system LAPACK for
the Cholesky factorization and inverse; the local `liblapack` on the development
machine is the reference implementation, so an optimized OpenBLAS/MKL link is
still the next CPU performance step.

Live mutations use exact algebra against the maintained inverse `A`:

* insertion — Schur-complement growth of `A`;
* label replacement — a rank-one update of `W` using the on-demand row
  `P_i = A[i,:] X`;
* deletion — the principal-submatrix inverse
  `A' = A_RR − A_Ri A_iR / A_ii`, one outer product, followed by a rank-one
  correction to `W`.

Cholesky is the factorization used to build `A` initially and to rebuild it
after a numerically unsafe downdate.  This version does not keep a Cholesky
factor synchronized across every mutation; `A` is the live exact state.

Deletion no longer rebuilds the solution.  The lexical novelty inverse is
updated through the same insertion/deletion path, so the first novelty query
after a write does not pay for a fresh factorization.  When no static encoder
is present, the lexical inverse is the full inverse and is reused directly.

The feature vocabulary and IDF weights stay fixed between fits.  Live writes
use only features already in that vocabulary; a new key can update weights but
cannot introduce new lexical coordinates.  This is why the incremental
identities are valid across live writes.  Attribution, counterfactual previews,
and pruning APIs are not implemented in this port.

The update is exact for the regularized ridge objective; classification
is still a learned prediction and can be affected by feature collisions.
Only the dictionary GET guarantees the exact stored value. Key identity
is case-sensitive even though lexical features are lowercased.

## Static embeddings (optional)

The static half of the feature map is a static embedding model, extracted
once from [model2vec](https://github.com/MinishLab/model2vec) into a tiny
dependency-free binary that the runtime reads:

```bash
make models-base          # or: make models-multilingual
```

That downloads from the Hugging Face hub and converts to semrember's own
binary format, in Crystal, verifying the result against a published
checksum. It is idempotent: if the file is present and its checksum
matches, nothing happens.

The encoder is **optional**. Without it semrember uses the TF-IDF byte
n-gram half alone: no encoder file, no download, no dependency — the
"floppy disk" mode, and the reason the library can be 68 KB next to a
2 MB binary.

The two halves behave differently, and the difference is the design
constraint the whole API is built around:

| | lexical (TF-IDF) | static (embeddings) |
|---|---|---|
| typos | survives | usually survives |
| paraphrase | misses | survives |
| out-of-distribution detection | **works** | **fails** |

That last row is why the novelty score is measured on the lexical half
alone: a static encoder will happily place nonsense next to a real
sentence, so including it collapses the separation between "unfamiliar"
and "out of distribution". With the static encoder present, it buys
paraphrase tolerance and pays for it in over-confidence — the novelty gate
is the counterweight.

See [`MODELS.md`](../MODELS.md) for the sources, the extraction, and the
footprint.

## Cross-implementation parity

`AMEM` is meant to be a portable interchange format, so this implementation
is checked against an independent one on three axes that must agree:

```
1. AMEM bytes   a model written by one is byte-identical to the other's
2. Scores       the same model and query give the same numbers
3. Token ids    the static encoder segments text identically
```

The original lexical comparison used the 42-signature incident corpus.
AMEM bytes were identical, the largest score difference was 8.1e-07, and
both implementations could load each other's files.

The historical static comparison used the companion Python semrember
implementation. That comparison did not verify parity with model2vec.
The regression suite now checks specific tokenizer cases against IDs from
the shipped models, including ASCII symbols, control and format characters,
and Greek sigma. WordPiece embeddings exclude unknown tokens from the
mean. The `tokens` CLI command shows the raw tokenizer IDs before pooling.

The `tools/parity.py` maintainer harness still compares with the companion
implementation. It requires a separate reference checkout, and the static
comparison can diverge when that checkout uses the older tokenizer behavior.

```bash
python3 tools/parity.py --python-root ../
```

The BERT tokenizer lowercases each codepoint independently, mapping
`"ΣΣ"` to `"σσ"`. The separate `Unicode.downcase` helper preserves the
contextual lowercase behavior used by the companion implementation.

The static encoder accumulates its token mean in float32. It keeps the
normalized vector in float64 until the caller casts it. Rounding the norm
early propagates into the fitted weights.

The Unicode tables come from the C implementation
(`tools/gen_unicode_tables.py`).

## Performance and footprint

Historical measurements from the original implementation, in-process after load.
These are not benchmarks of the current Kemal HTTP service:

| | Crystal | Python reference |
|---|---:|---:|
| query (lexical, `d=2127`) | **2.1 µs** | 11.4 µs |
| `semrember` CLI | 2.2 MB | — |
| `semrember-server` | 4.3 MB | — |
| 42-signature brain (`AMEM`) | 68 KB | 68 KB |
| base-8M encoder: load / resident | 0.01 s / 20 MB | 0.39 s / 115 MB (mmap) |
| multilingual-128M encoder: load / resident | 0.27 s / 327 MB | 0.39 s / 115 MB (mmap) |

The Crystal encoder *reads* the model; the C library *memory-maps* it,
which is why the 253 MB multilingual model costs 327 MB here and 115 MB
there. If resident memory matters more than the last millisecond of load
time, use the 15 MB model or the C library.

### Incremental writer measurements

Synthetic sparse lexical workload, `N=850`, `D=7,523`, `T=20`, `min_df=2`,
on the development machine (`make bench-writer`):

| operation | before | after |
|---|---:|---:|
| `fit` | 23.5 s | **1.6 s** |
| first novelty | 3.6 s | **1 ms** |
| delete one key | 22.9 s | **6.8 ms** |
| delete 20 keys | 23.3 s | **95 ms** |

The change is algorithmic, not a micro-optimization: sparse Gram construction,
Cholesky instead of Gauss-Jordan, `W = (AY)ᵀX` instead of materialising `AX`,
removal of the dense `X` cache, principal-submatrix deletion, and an
incrementally maintained novelty inverse.  AMEM bytes on the bundled incident
corpus are unchanged.

The engine is ~2,800 lines of Crystal plus ~2,600 generated table lines;
the whole public repository is under 4,000 hand-written lines.

## Repository layout

```
src/semrember.cr        the engine: features, writers, AMEM I/O
src/codec.cr            little-endian primitives
src/unicode.cr          normalization shared with the C implementation
src/unicode_tables.cr   generated — do not edit
src/tokenizer.cr        WordPiece and SentencePiece-Unigram
src/static_encoder.cr   in-memory encoder for the static models
src/server.cr           the Kemal routes and HTTP/JSON service
src/server_cli.cr       the service executable entry point
src/influence.cr        leave-one-out decision power and pruning
src/cli.cr              the command line front end
src/csv.cr              RFC 4180 reader
spec/                   engine, tokenizer, persistence and HTTP regressions
clients/python/         the Python client
tools/                  table generator, model fetcher, parity harness, benchmarks,
                        Banzhaf Monte Carlo, retention experiment
```

## Format

The on-disk format is `AMEM` v3 — see [`FORMAT.md`](../FORMAT.md). Files are
little-endian, portable between the Crystal, Python and C implementations,
and versioned; a newer format is rejected rather than misread.

## Limitations

* **A loaded model has no training keys.** `AMEM` stores weights and
  vocabulary, not the training set. So a model loaded from disk can
  classify, but cannot compute similarity or novelty — those need the live
  observations. The server's `--train` mode keeps them; `--model` does not.
* **An empty live dual writer bootstraps on its first write.** After
  deleting its last key, it can bootstrap again. Loaded AMEM readers cannot
  accept semantic updates: service writes report `semantic_updated: false`,
  and semantic endpoints return HTTP 409 until the service is started with
  the original corpus using `--train`.
* **The vocabulary is fixed at fit time.** A new observation cannot
  introduce new n-grams; a refit changes every encoding.
* **Predictions can interfere.** Capacity depends on feature separability,
  regularization, and the data; the feature dimension is not a guarantee
  that every training association will be predicted correctly.
* **Redis compatibility is a mental model, not a wire protocol.** The
  transport is HTTP+JSON and the command names are borrowed so the concepts
  transfer. This is not the RESP protocol and does not claim to be.
* **This is a first responder, not an autopilot.** The semantic layer
  returns candidates and confidence; the caller decides.

## License

MIT.
