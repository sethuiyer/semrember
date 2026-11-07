# Static embedding models

The static half of semrember's feature map is a **static embedding model**:
text is tokenised, token vectors are pooled, and the result is one dense
vector. It is what buys synonymy and paraphrase tolerance; it is also what
makes a model *confidently wrong* on unfamiliar input, because a static
encoder maps any string — including nonsense — into a plausible point of
its embedding space.

## Where they come from

The models are distilled from sentence transformers by
[model2vec](https://github.com/MinishLab/model2vec) (MIT), by Minish Lab
(Stephan Tulkens, Thomas van Dongen and others). They are **not**
redistributed in this repository; download or extract them yourself.

| model | tokenizer | vocab | dim | file |
|---|---|---:|---:|---:|
| [`minishlab/potion-base-8M`](https://huggingface.co/minishlab/potion-base-8M) | WordPiece (v1) | 29,528 | 256 | 15.4 MB |
| [`minishlab/potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M) | SentencePiece Unigram (v2) | 500,353 | 256 | 265.0 MB |

## Getting them

semrember does not read model2vec's own format at runtime. The models are
re-serialised once into a tiny, dependency-free binary format — `M2VE` for
WordPiece, `M2VS` for SentencePiece — and that is what the engine reads.

The fetcher below does the whole job in Crystal: it downloads from the
Hugging Face hub, converts, and verifies the result against a published
checksum. It needs no Python, no model2vec, and no tokenizers library.

```bash
make models-multilingual        # or: make models-base
```

```bash
crystal run --release tools/fetch_model.cr multilingual
# fetching minishlab/potion-multilingual-128M
#   tokenizer.json: 17 MB
#   model.safetensors: 488 MB
# extracting potion-multilingual-128M.bin
# wrote models/potion-multilingual-128M.bin (252 MB, sha256 ok)
```

It is idempotent: if the file is already there **and its checksum matches**,
nothing is downloaded. A checksum mismatch is a hard failure — the file is
deleted rather than silently installed.

```bash
tools/fetch_model.cr --list          # what is available
tools/fetch_model.cr base --force    # refetch
```

A full cold run for the multilingual model — download 512 MB, convert
128M float32 weights to float16, verify — takes about **28 seconds**.

The result is byte-identical to the reference `potion-multilingual-128M.bin`,
which is what makes the checksum a meaningful test rather than a receipt.

### With model2vec instead

The original extractor is kept for reference, and needs Python and
model2vec:

```bash
pip install model2vec
python tools/extract_static.py minishlab/potion-base-8M            models/potion-base-8M.bin
python tools/extract_static.py minishlab/potion-multilingual-128M models/potion-multilingual-128M.bin
```

Both paths produce the same bytes.

The `M2VS` v2 format stores the Unigram log-probabilities, so the runtime
can run the same Viterbi segmentation as the source tokenizer. WordPiece
models are lowercased and accent-stripped exactly like BERT.

The file format is documented in [`FORMAT.md`](FORMAT.md).

## Using one

```bash
build/semrember --static models/potion-base-8M.bin train corpus.csv message action brain.amem
build/semrember-server --train corpus.csv --key message --value action --static models/potion-base-8M.bin
```

```crystal
enc = Semrember::StaticEncoder.new("models/potion-base-8M.bin")
mem = Semrember::Memory.new(solver: :dual, static: enc)
```

```python
# over HTTP, from any language
kb = Semrember("http://127.0.0.1:7379")
kb.similar("disk full")
```

## The lexical half needs none of this

The encoder is **optional**. Without it, semrember uses the TF-IDF byte
n-gram half alone: no encoder file, no download, no dependency — which is
the "floppy disk" mode, and the reason the library can be 68 KB with a
532 KB binary.

The two halves behave differently, and the difference is a design
constraint rather than an accident:

| | lexical (TF-IDF) | static (embeddings) |
|---|---|---|
| typos | survives (character n-grams) | usually survives |
| paraphrase | misses | survives |
| out-of-distribution detection | **works** | **fails** |

That last row is why the novelty score is measured on the lexical half
alone. Including the static half collapses the separation between
"unfamiliar" and "out of distribution" — the static encoder will happily
place nonsense next to a real sentence. When the static encoder is present,
it buys paraphrase tolerance and pays for it in over-confidence; the
novelty gate is the counterweight.

## Footprint

Measured on this machine:

| model | file | load | resident |
|---|---:|---:|---:|
| `potion-base-8M` | 14.7 MB | 0.01 s | 20 MB |
| `potion-multilingual-128M` | 252.8 MB | 0.27 s | 327 MB |

The Crystal encoder **reads** the model into memory. The C implementation
**memory-maps** it, which is why the same 253 MB file costs ~115 MB there
— the pages are file-backed and shared. If resident memory matters more
than the last millisecond of load time, use the C library, or use the
15 MB model.

```bash
MODEL=models/potion-multilingual-128M.bin crystal run --release tools/encoder_stats.cr
```

## Credits

```bibtex
@software{minishlab_model2vec,
  author    = {Tulkens, Stephan and van Dongen, Thomas and others},
  title     = {Model2Vec: Turn any Sentence Transformer into a Small Static Model},
  year      = {2024},
  url       = {https://github.com/MinishLab/model2vec},
  license   = {MIT}
}
```

The models are distilled from sentence transformers using model2vec's
distillation pipeline. semrember only re-serialises the vocabulary and the
embedding matrix.
