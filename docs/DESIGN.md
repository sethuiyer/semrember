# Engine design and historical measurements

[Start with the developer quickstart](QUICKSTART.md).

## The writers

| solver | state | semantics |
|---|---|---|
| `delta` | `W` | iterative sweeps; the large-`D` fallback |
| `rls` | `P` (`D×D`) | exact ridge, recursive least squares |
| `dual` | `A` (`N×N`) | the same exact ridge solution in the kernel dual |

`dual` is the default because it is cheaper whenever `N < D` — the regime
this library targets — and because it supports **exact replacement**:

```
A₋ᵢ = A + u uᵀ / t,     t = 1 − xᵢᵀ A xᵢ
```

is a rank-1 downdate, so removing or replacing one observation is algebra
rather than treating a replacement as a second observation. This port
uses a rank-1 target update for replacement and a Schur-complement update
for insertion. Deletion rebuilds the solution from retained observations.
Attribution, counterfactual previews, and pruning APIs are not implemented
in this port.

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

Measured on the 42-signature incident corpus:

| | lexical | base-8M | multilingual-128M |
|---|---|---|---|
| AMEM bytes identical | **yes** | 1 of 21,270 float16 weights differs by 1 ULP | 2 of 21,270 differ by 1 ULP |
| score agreement | 8.1e-07 | 5.8e-07 | 9.3e-07 |
| token ids identical | — | **yes** (WordPiece) | **yes** (SentencePiece Unigram + Viterbi) |
| cross-load both ways | **yes** | **yes** | **yes** |

The token-id check covers the awkward cases — Greek final sigma, combining
marks, Devanagari, CJK, and punctuation isolation — because those are
exactly where two independent implementations of the same tokenizer drift
apart.

The one differing weight is a float16 rounding boundary: the two
implementations accumulate the static encoder's normalisation sum in a
different order, which is worth ~1e-7 relative, which lands on the wrong
side of a rounding tie for one value in ten thousand.

The harness that produces these numbers is `tools/parity.py`. It is a
**maintainer tool**: it compares against a reference implementation that is
not part of this repository, so a public checkout cannot reproduce it. It is
kept here because a conformance claim is only worth something if the check
that backs it is visible:

```bash
python3 tools/parity.py --python-root ../ --static models/potion-base-8M.bin
```

Matching the reference bit-for-bit required reproducing two details that are
easy to get wrong and impossible to guess:

* **the contextual final-sigma rule** — Python's `str.lower()` maps `"ΣΣ"` to
  `"σς"`; a codepoint-wise mapping gives `"σσ"`. Crystal's `String#downcase`
  is codepoint-wise, so the rule is implemented explicitly here, using the
  same cased and case-ignorable tables the C library uses.
* **the type flow of the arithmetic** — the reference keeps the static vector
  in float64 until the caller casts it, and computes the joint norm over
  that float64 value. Rounding the norm early propagates into every
  weight.

The Unicode tables are generated from the C implementation
(`tools/gen_unicode_tables.py`) rather than re-derived, so the two agree by
construction rather than by luck.

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
src/cli.cr              the command line front end
src/csv.cr              RFC 4180 reader
spec/                   engine, tokenizer, persistence and HTTP regressions
clients/python/         the Python client
tools/                  table generator, model fetcher, parity harness, benchmark
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
