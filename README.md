# semrember

A compressed associative memory in Crystal: text → a learned label matrix,
read with one matrix-vector product.

* **One binary, no runtime dependencies.** C11 + libc is the floor; Crystal
  compiles to a single static executable.
* **Exact, not approximate.** The dual writer gives a *hard* replacement: a
  key maps to one value, and overwriting it replaces the value rather than
  quietly adding a second observation.
* **A Redis-shaped API, with a semantic layer Redis does not have.** The
  exact operations (`GET`, `SET`, `DELETE`, `EXISTS`, `SCAN`) never invoke
  the model. The semantic operations (`SIMILAR`, `ESCALATE`, `NOVELTY`)
  always return candidates and confidence, never a bare answer.
* **The file format is the contract.** `AMEM` v3 is documented in
  [`FORMAT.md`](FORMAT.md) and versioned: anything that reads and writes
  those bytes produces and consumes the same models.

```
key → repr() → [TF-IDF byte n-grams ‖ static embedding] → joint L2
              → W @ q → argmax label
```

---

## Build

```bash
make            # builds build/semrember and build/semrember-server
make test       # 20 examples
```

Nothing else is needed: no shards, no C libraries, no runtime packages.

## Quickstart

### As a service

```bash
build/semrember-server --train examples/incidents.csv --key message --value action
# semrember 1.0.0 listening on http://127.0.0.1:7379
```

```bash
curl localhost:7379/v1/kv/"No%20space%20left%20on%20device"
# {"key":"No space left on device","value":"free_disk"}

curl -X POST localhost:7379/v1/similar \
     -H 'Content-Type: application/json' \
     -d '{"text":"disk full","top_k":3}'
# {"candidates":[{"key":"filesystem is full","value":"free_disk","score":0.346}, ...]}
```

### From Python

The client is standard library only — no `requests`, no `redis-py`, no numpy.

```python
from semrember_client import Semrember

kb = Semrember("http://127.0.0.1:7379")

kb.set("No space left on device", "free_disk")   # -> previous value
kb.get("No space left on device")                # -> "free_disk"
kb.exists("No space left on device")             # -> True
kb.delete("No space left on device")             # -> True

for cand in kb.similar("disk full", top_k=3):
    print(cand["key"], cand["value"], cand["score"])

d = kb.escalate("ImagePullBackOff")
# {"decision": "escalate", "reason": "both", "margin": -0.646, "novelty": 0.734, ...}
```

See `clients/python/semrember_client.py`.

### As a library

```crystal
require "./src/semrember"

mem = Semrember::Memory.new(solver: :dual)
mem.fit({"No space left on device" => "free_disk",
         "could not resolve host api" => "fix_dns"})

mem.query("disk full")            # => [{"free_disk", 0.42}]
mem.set("disk full", "clear_cache")   # exact replacement
mem.similar("no space left", 3)  # nearest stored keys
mem.novelty("zqxwvbnmwk")         # => 0.93  (out of distribution)
mem.save("brain.amem")
```

### As a CLI

```bash
build/semrember train data.csv message action brain.amem 1 dual
build/semrember query brain.amem "no space left on device" 3
build/semrember scores brain.amem "no space left on device"   # JSON, all labels
build/semrember tokens models/potion-base-8M.bin "hello world"
```

---

## The API, and why it is split

Two families of operation, and the split is the point.

| | exact | semantic |
|---|---|---|
| endpoints | `GET` `PUT` `DELETE` `EXISTS` `SCAN` `EXPIRE` `TTL` | `SIMILAR` `ESCALATE` `NOVELTY` |
| touches the model? | **never** | yes |
| returns | the stored value, or 404 | candidates, scores, confidence |
| failure mode | absent key | ambiguous vs out-of-domain, named |

`GET` is a hash lookup. It does not consult the weights, so it cannot be
changed by a retrain, a correction, or a threshold. That is what makes it
safe to build a cache on top of it.

`ESCALATE` is the part Redis has no equivalent for. It applies **two
independent gates** and names which one fired:

```json
{"decision": "escalate", "reason": "out_of_domain",
 "margin": 0.986, "novelty": 0.734,
 "candidates": [{"label": "free_disk", "score": 0.95}, ...]}
```

* **margin** — `(s1−s2)/(|s1|+|s2|)`, the scale-free top-2 gap. Catches
  *ambiguity*: the input is in the taxonomy but two actions are close.
* **novelty** — `1 − k'(XX'+λI)⁻¹k`, the ridge leverage on the lexical
  subspace. Catches *out-of-domain*: nothing stored resembles this input.

They are correlated at r ≈ −0.7, yet they fail on **different** inputs: a
confidently-wrong routing needs the novelty gate, and a familiar-but-wrong
one needs the margin gate. A single threshold conflates two different
reasons to escalate, and the reasons call for different people.

Full contract: [`API.md`](API.md).

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
rather than a retrain. That single identity is what makes a key-value store
on top of this model honest, and it is what `semrember.prune` certifies
decisions with.

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

See [`MODELS.md`](MODELS.md) for the sources, the extraction, and the
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

Measured on this machine, in-process, after load:

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
src/static_encoder.cr   mmap-style encoder for the static models
src/server.cr           the HTTP/JSON service
src/cli.cr              the command line front end
src/csv.cr              RFC 4180 reader
spec/                   17 examples
clients/python/         the Python client
tools/                  table generator, model fetcher, parity harness, benchmark
```

## Format

The on-disk format is `AMEM` v3 — see [`FORMAT.md`](FORMAT.md). Files are
little-endian, portable between the Crystal, Python and C implementations,
and versioned; a newer format is rejected rather than misread.

## Limitations

* **A loaded model has no training keys.** `AMEM` stores weights and
  vocabulary, not the training set. So a model loaded from disk can
  classify, but cannot compute similarity or novelty — those need the live
  observations. The server's `--train` mode keeps them; `--model` does not.
* **The vocabulary is fixed at fit time.** A new observation cannot
  introduce new n-grams; a refit changes every encoding.
* **Capacity is bounded by the feature dimension**, roughly `D`. Past that
  the matrix saturates and predictions interfere.
* **Redis compatibility is a mental model, not a wire protocol.** The
  transport is HTTP+JSON and the command names are borrowed so the concepts
  transfer. This is not the RESP protocol and does not claim to be.
* **This is a first responder, not an autopilot.** The semantic layer
  returns candidates and confidence; the caller decides.

## License

MIT.
