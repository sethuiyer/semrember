# AMEM v3 — model file format

The on-disk format for a trained semrember model. Portable between the
Crystal, Python and C implementations: a file written by any of them loads
in the others, and the reference implementations produce byte-identical
files.

All integers and floats are **little-endian**.

---

## Layout

```
offset  size        contents
------  ----------  ---------------------------------------------------
  0     4           magic, ASCII "AMEM"
  4     4           format version, uint32 (currently 3)
  8     4           D_tfidf, uint32  — vocabulary size
 12     4           D_static, uint32  — static embedding dimension (0 = none)
 16     4           T, uint32         — number of labels
 20     4           min_df, uint32   — document-frequency floor used at fit
 24     4           bloom_bytes, uint32
 28     bloom_bytes  Bloom filter over the vocabulary
  ...   variable    D_tfidf term entries
  ...   variable    T label entries
  ...   T*(Dt+Ds)*2 weight matrix, float16, row-major
```

### Term entries

Repeated `D_tfidf` times, in vocabulary-index order:

```
  uint16  length in bytes
  bytes   the term (UTF-8; byte n-grams, so not necessarily text)
  float32 IDF weight
```

The vocabulary is sorted by byte value, so index order is deterministic.

### Label entries

Repeated `T` times, in label-index order:

```
  uint16  length in bytes
  bytes   the label, UTF-8
```

### Weights

`T × (D_tfidf + D_static)` float16 values, **row-major**: label `c`,
feature `d` lives at offset `(c * (D_tfidf + D_static) + d) * 2`.

float16 is IEEE 754 binary16, round-to-nearest-even. The three
implementations agree bit for bit on the conversion.

---

## What is *not* stored

The format stores a model, not a training set:

* **the training keys and their targets** — so a loaded model can classify
  but cannot compute similarity or novelty, which need the live
  observations;
* **the writer state** (`P` for RLS, `A`/`X` for the dual writer) — so a
  loaded model cannot accept exact writes;
* **pumping or repetition observations**;
* **the static embedding model** — `D_static` records only its dimension,
  and the file must be opened with a matching encoder.

A model loaded from disk is therefore a **reader**. Continuing to train
requires the original corpus.

---

## The Bloom filter

The header carries a Bloom filter over the vocabulary. It is not required
to *read* a model — it is a membership accelerator — but its size and
contents are fixed by the format, so a writer must reproduce it exactly
or the files will not be byte-identical.

The layout is deterministic:

```
n_bits = 8
while n_bits < 15 * number_of_terms:  n_bits *= 2
bits   = n_bits / 8 bytes, zeroed
for each term, for each of the four seeds:
    h = fnv1a(term, seed) & (n_bits - 1)
    bits[h >> 3] |= 1 << (h & 7)
```

with the seeds `2166136261, 2246822519, 3266489917, 668265263` and FNV-1a
(32-bit) as the hash.

---

## Versioning

The version field is checked on load. An unknown version is **rejected**,
not guessed at. A future format that changes the weight layout or the
semantics of a field will get a new version number.

The current version is **3**.

---

## Worked example

```bash
build/semrember train corpus.csv message action brain.amem 1 dual
build/semrember info brain.amem
```

```crystal
mem = Semrember::Memory.load("brain.amem")
mem.query("no space left on device")   # => [{"free_disk", 0.42}]
```

The format is the interop contract: anything that reads and writes these
bytes produces and consumes the same models. `tools/parity.py` is the
conformance harness used to check an independent implementation against
this one — see its header for what it requires.
