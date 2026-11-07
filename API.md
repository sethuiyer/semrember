# semrember service API

Version `v1`. Transport is HTTP+JSON; every response is a JSON object.
Errors carry a non-2xx status and an `error` string.

The command names are borrowed from Redis so the concepts transfer. The
protocol is not RESP and does not claim Redis compatibility.

---

## Exact operations

These never invoke the model. A `GET` is a hash lookup: it cannot be
changed by a retrain, a correction, or a threshold.

### `GET /v1/kv/{key}`

```json
{"key": "No space left on device", "value": "free_disk"}
```

`404` with `{"error": "not found"}` when the key is absent. Keys are
percent-decoded; a key may contain any byte sequence.

### `PUT /v1/kv/{key}`

Body: `{"value": "free_disk"}`

```json
{"ok": true, "previous": "free_disk"}
```

`previous` is `null` for a new key. Replacing a key **replaces** its value:
the model is updated with an exact rank-1 overwrite, so neighbouring
records move as little as the algebra allows. It does not add a second
observation for the same key.

### `DELETE /v1/kv/{key}`

```json
{"deleted": true}
```

`false` when the key was absent. Deletion is idempotent.

### `GET /v1/kv/{key}/exists`

```json
{"exists": true}
```

### `GET /v1/kv?cursor=0&match=*&count=100`

A deterministic scan over sorted keys.

```json
{"cursor": "0", "keys": ["a", "b", "c"]}
```

`cursor` is `"0"` when the scan is complete. `match` accepts `*` and `?`
wildcards. `count` bounds the page size (default 100).

### `POST /v1/kv/{key}/expire`

Body: `{"seconds": 60}`

```json
{"ok": true}
```

### `GET /v1/kv/{key}/ttl`

```json
{"ttl": 59}
```

`ttl` is `null` when the key has no expiry or does not exist.

---

## Semantic operations

These invoke the model. They return **candidates and confidence**, never a
bare answer: several records may share one label, so a single value would
be a lie.

### `POST /v1/similar`

Body: `{"text": "disk full", "top_k": 3, "min_score": 0.1}`

```json
{"candidates": [
  {"key": "filesystem is full", "value": "free_disk", "score": 0.346},
  {"key": "disk quota exceeded", "value": "free_disk", "score": 0.288}
]}
```

Records most similar to the text, ranked by cosine over the same features
the model uses. `min_score` is optional.

### `POST /v1/escalate`

Body: `{"text": "...", "min_margin": 0.35, "min_novelty": 0.55}`

```json
{
  "decision": "escalate",
  "handled": false,
  "reason": "out_of_domain",
  "margin": 0.986,
  "novelty": 0.734,
  "action": null,
  "candidates": [{"label": "free_disk", "score": 0.95}]
}
```

Two independent gates decide, and `reason` names which one fired:

| reason | meaning |
|---|---|
| `handled` | act on `action` |
| `ambiguous` | in the taxonomy, but two actions are close — a human picks |
| `out_of_domain` | nothing stored resembles this — the taxonomy has a gap |
| `both` | unfamiliar *and* ambiguous |

* **margin** — `(s1−s2)/(|s1|+|s2|)`, the scale-free top-2 gap, in `[-1, 1]`.
* **novelty** — `1 − k'(XX'+λI)⁻¹k`, the ridge leverage on the lexical
  subspace, in `[0, 1]`.

The two are correlated at r ≈ −0.7 but fail on different inputs: a
confidently-wrong routing needs the novelty gate, a familiar-but-wrong one
needs the margin gate. `min_margin` and `min_novelty` are dataset-specific
and are not calibrated defaults.

### `GET /v1/novelty?text=...`

```json
{"novelty": 0.734}
```

---

## Server

### `GET /v1/ping`

```json
{"ok": true, "version": "1.0.0"}
```

### `GET /v1/info`

```json
{
  "version": "1.0.0",
  "uptime_seconds": 12,
  "records": 42,
  "labels": 10,
  "d_tfidf": 2127,
  "d_static": 0,
  "solver": "dual",
  "exact_endpoints": ["GET", "PUT", "DELETE", "EXISTS", "SCAN"],
  "semantic_endpoints": ["SIMILAR", "ESCALATE", "NOVELTY"]
}
```

---

## Durability and consistency

The service is **in-memory**. Records live in the process; there is no
write-ahead log, no persistence, and no replication. Restarting loses
everything that was not written to a model file.

Within a single process:

* Reads are safe concurrently.
* Writes are serialised by one lock, so a `PUT` is atomic with respect to
  other operations.
* The record store is the source of truth for exact operations. The model
  is a secondary index over it.

That last point is the important one: if the model cannot accept a write —
for example a model loaded from disk, whose writer state was dropped — the
exact record store still holds the value, and `GET` still returns it. The
model catches up when it can.

## Deployment

```bash
semrember-server --train corpus.csv --key message --value action
semrember-server --model brain.amem --static models/potion-base-8M.bin
semrember-server --host 0.0.0.0 --port 7379
```

**There is no authentication.** The service binds to `127.0.0.1` by default.
Put it behind something that authenticates before exposing it beyond the
host — it has no notion of tenants or namespaces.
