# What Semrember remembers

[README](../README.md) · [Quickstart](QUICKSTART.md) · [Binary format](../FORMAT.md)

## Exact records and learned associations

The service has two kinds of memory. The **record dictionary** keeps the exact
string key and string value. The **learned model** maps features of input text
to scores over the labels it was trained on. The live writer also retains
observations and matrices needed for updates and semantic reads.

For `"certificate has expired" → "renew_certificate"`:

- GET with the exact key returns the stored string `renew_certificate`.
- SIMILAR retrieves related retained keys and their current record values.
- ESCALATE scores possible labels and applies ambiguity and novelty gates.
- A CLI/library query ranks labels directly; it does not apply escalation gates.

Exact keys are case-sensitive. Lexical features lowercase ASCII, so two keys
such as `Disk` and `disk` can be distinct dictionary records while having
identical lexical features. Their exact values can differ; the model cannot
reliably distinguish identical features with conflicting labels.

## What it can learn

Semrember can learn associations between text patterns and labels you supply.
Several keys can share a label. Byte n-grams capture overlapping local text
patterns and can tolerate some spelling changes. Optional static embeddings
add features that can help with paraphrases and synonyms.

Values are whole string labels, not executable actions or arbitrary JSON
objects. The service never executes the action name it returns. A long value
is still one categorical label, not a document whose individual facts become
independently queryable. AMEM label strings must fit in 65,535 bytes.

## What it cannot promise

| Expectation | Actual behavior |
|---|---|
| Every training association is always predicted perfectly | Ridge regularization, feature collisions, and interference can affect predictions |
| A score of 0.9 means 90% confidence | Scores are raw model outputs, not calibrated probabilities |
| An unrelated query has no answer from `query()` | A nonempty model still ranks its labels; use escalation when rejection matters |
| It remembers every word in a document independently | It learns a text-to-label association using the encoded features |
| It reasons about negation, chronology, or conversation history | There is no general reasoning engine or automatic conversational state |
| It generates a new answer | It scores supplied labels or retrieves retained examples |
| More keys always mean more usable memory | Useful capacity depends on separable features and the data, not just record count |
| A small AMEM file is the whole live system's memory footprint | Live observations, inverse matrices, and an optional encoder use additional RAM |

SIMILAR and ESCALATE provide evidence for the application, not a correctness
guarantee. Evaluate routing on held-out examples and tune `min_margin` and
`min_novelty` for your own data. The default thresholds are starting values.
The lexical novelty score can also mistake shared spelling for familiarity;
it is not a universal out-of-distribution detector.

## What happens when new keys arrive

`fit()` builds the n-gram vocabulary and IDF weights. They stay fixed during
subsequent writes. A new key can update the label matrix through features
already in that vocabulary, plus the optional static encoder's features.
It cannot introduce new lexical features without a refit.

An exact PUT can therefore succeed while the new key has an inadequate
representation for semantic prediction. GET still returns its exact value.
`semantic_updated: true` means the writer accepted the observation; it is not
a promise that the new key will be classified correctly.

An empty live service bootstraps its vocabulary from its first write. After
the last observation is deleted, another write can bootstrap it again. For a
real application, starting with a representative corpus is preferable to
building the vocabulary from one short key.

The live `dual` solver supports `set()` and `delete()`. The library's `delta`
and `rls` solvers are fitting alternatives; they do not expose the dual
writer's exact update operations.

## What survives a save

AMEM v3 is a **classification reader format**, not a service snapshot.

| State | Saved in AMEM? |
|---|---|
| Vocabulary and IDF | Yes |
| Label strings | Yes |
| Prediction weights | Yes, quantized to float16 |
| Training keys and their assignments | No |
| Live observation features and writer inverse matrices | No |
| The service's exact record dictionary | No |
| TTL deadlines | No |
| Static encoder vocabulary/embedding file | No; retain it separately |

Label strings survive, but the associations needed to recover the original
exact key/value dictionary do not. Saved scores can differ slightly from live
float32 scores because the file uses float16 weights.

## Save a classification reader

From the repository root:

```bash
mkdir -p build
build/semrember train examples/incidents.csv message action build/brain.amem
build/semrember query build/brain.amem "certificate has expired" 3
build/semrember info build/brain.amem
```

The first ranked label for this example should be `renew_certificate`.
From Crystal, use `memory.save(path)` and `Semrember::Memory.load(path)`;
see [the runnable save/load example](../examples/quickstart.cr).

The CLI or library can classify with the loaded reader. `Memory.load()` cannot
resume `set()`, `delete()`, `similar()`, or `novelty()` from that file. Refit it
with the source corpus if you need live observations again.

For a hybrid model, pass the same `--static` encoder path at training and
query/load time. Missing or wrong-dimension encoders are rejected. AMEM v3
cannot distinguish two different encoders of the same dimension, so retaining
the original encoder is the caller's responsibility.

Retrain hybrid AMEM files built before the static tokenizer correction.
Their weights were fitted with the earlier feature mapping, and AMEM v3
cannot identify which tokenizer behavior produced them.

## Starting a service from AMEM

```bash
build/semrember-server --model build/brain.amem --port 7380
```

| Operation on this service | Result |
|---|---|
| GET of an original training key | HTTP 404; the exact store starts empty |
| PUT of a new exact record | Accepted, with `semantic_updated: false` |
| GET of that newly written exact key | Returns its in-memory string value |
| SIMILAR, NOVELTY, ESCALATE | HTTP 409 because observations are unavailable |
| Restart again with the same AMEM | Exact store is empty again |

`/v1/info` reports `semantic_reads: false` and `semantic_updates: false`.
There is no automatic background catch-up. There is also no separate HTTP
classification endpoint for the loaded weights; use the CLI/library, or start
with `--train` to use the live semantic API. `--train` and `--model` are
mutually exclusive.

## Durable application data

By default, a successful PUT is an in-memory acknowledgement. It does not write
back to the input CSV or an AMEM file. For built-in encrypted record snapshots,
use [durable mode](DURABILITY.md). It restores records, absolute TTLs and a rebuilt
writable semantic index. There is no replication or HTTP save endpoint.

To recover application state managed externally instead:

1. Retain current records and accepted changes in your application's durable store.
2. Remove deleted/expired records and export the current keys and labels as a corpus.
3. Restart with `--train` and the appropriate `--key`/`--value` column names.
4. If expiry matters, reapply remaining TTLs from your durable deadlines. CSV has no TTL field.

Refitting rebuilds vocabulary and IDF; semantic scores can change compared
with the earlier model whose online vocabulary stayed fixed. The current
exact record values come from your restored source records.

Expiration is lazy. A read or semantic operation removes expired observations
from the relevant live state before using them. A PUT replaces a record and
clears its prior TTL. In default mode, neither deadlines nor live writes survive a restart. Durable
mode preserves them and removes expired records before rebuilding the index.

## Cost of live memory

The dual writer retains observation-space state that grows quadratically in
the number of observations. Deletion rebuilds the solution. The first novelty
query after an observation change builds and inverts a lexical Gram matrix;
later queries reuse that inverse. A label-only overwrite does not require
rebuilding the novelty inverse. Benchmark with your corpus and read/write mix.
