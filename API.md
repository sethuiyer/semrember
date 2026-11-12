# semrember service API

Version `v1`. Kemal serves the HTTP+JSON API; every response is a JSON object
(except HEAD responses, which have no body).
Errors carry a non-2xx status and an `error` string.

The command names are borrowed from Redis so the concepts transfer. The
protocol is not RESP and does not claim Redis compatibility.

For a runnable walkthrough, start with the [five-minute quickstart](docs/QUICKSTART.md).
See [API examples](docs/EXAMPLES.md) for curl, Python and JavaScript, and
[memory and persistence](docs/MEMORY.md) for the live-service/AMEM distinction.

---

## Exact operations

Reads return dictionary values. Writes also update the live semantic index.
A `GET` is a hash lookup: it cannot be
changed by a retrain, a correction, or a threshold.

### `GET /v1/kv/{key}`

```json
{"key": "No space left on device", "value": "free_disk"}
```

`404` with `{"error": "not found"}` when the key is absent. Keys are
percent-decoded. Percent-encode the entire key as one path segment,
including `/` as `%2F`. This distinguishes a literal key such as
`a%2Fexists` from the EXISTS operation at `a/exists`. JSON string keys
and values must be valid UTF-8.

### `PUT /v1/kv/{key}`

Body: `{"value": "free_disk"}`

```json
{"ok": true, "previous": "free_disk", "semantic_updated": true}
```

`previous` is `null` for a new key. Key identity is case-sensitive.
The first write into an empty live service bootstraps its vocabulary.
`semantic_updated` is false for a service started with an AMEM reader;
exact writes still work, but no background semantic catch-up occurs.
Replacing a key in a live dual writer **replaces** its value:
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
wildcards over Unicode characters, matching the entire key. `count`
bounds the page size (1–10000, default 100). An exhausted or oversized
cursor returns an empty page and cursor `"0"`; negative cursors are rejected.

### `POST /v1/kv/{key}/expire`

Body: `{"seconds": 60}`

```json
{"ok": true}
```

### `GET /v1/kv/{key}/ttl`

```json
{"ttl": 59}
```

`ttl` is `null` when the key has no expiry or does not exist. Expiry is
applied lazily before exact and semantic reads and removes both the
record and its live semantic observation. Expired records cannot be
revived by EXPIRE.

---

## Semantic operations

These invoke the model and return candidates, raw scores, and decision
heuristics. Scores are not probabilities or guarantees. Several records
may share a label; use SIMILAR when you need evidence tied to stored keys.

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
| `handled` | both gates passed; `action` is a suggestion for the application |
| `ambiguous` | the top-label margin is below the configured threshold |
| `out_of_domain` | lexical novelty exceeds the configured threshold |
| `both` | both gates failed; also used when the live service has no candidates |

* **margin** — `(s1−s2)/(|s1|+|s2|)`, the scale-free gap between the two highest scores, in `[0, 1]`.
* **novelty** — `1 − k'(XX'+λI)⁻¹k`, the ridge leverage on the lexical
  subspace, in `[0, 1]`.

The gates measure different properties and can still accept a wrong answer.
Choose `min_margin` and `min_novelty` using held-out examples from your own
data. Their defaults are starting values, not calibrated guarantees.

### `GET /v1/novelty?text=...`

```json
{"novelty": 0.734}
```

### `POST /v1/influence`

Body: `{"queries": ["disk full", "cert expired"], "max_examples": 3}`

Returns empirical leave-one-out pivotal power for each stored memory. This is
not the classical coalition Banzhaf index.

```json
{"memories": [
  {"key": "disk full", "index": 0, "beta": 0.142857, "pivotal_queries": 1,
   "mean_influence": 0.1377, "max_influence": 0.7084,
   "a_ii": 2.3586, "unstable": false, "examples": ["cache stale"]}
]}
```

### `POST /v1/compact`

Body:

```json
{"queries": ["disk full", "cert expired"], "beta_threshold": 0.0,
 "coverage_floor": 0, "pin": ["injury-claim"], "keep_per_label": 1,
 "verify": true}
```

Verified counterfactual compaction. Candidate batches are deleted on a copy
and accepted only if every query decision is preserved; failed batches are
split recursively. Exact KV records are unchanged. Compaction is not persisted
in durable mode: restart refits the semantic index from the records.

Returns `200` with counts when no query decision changed, or `409` when the
requested compaction could not be verified.

### Poison checks on `PUT`

`PUT /v1/kv/{key}` accepts two optional body fields:

```json
{"value": "free_disk", "poison_check": true, "reject_suspicious": false}
```

`poison_check` scores the proposed write against the rolling query buffer
(the last 128 SIMILAR / ESCALATE / NOVELTY inputs). For a new key the
counterfactual is absence -> insertion; for an existing key it is old value ->
proposed overwrite. The query buffer is caller-influenced and is governance
telemetry, not trusted security state. The response includes:

```json
{"poison_check": {
  "suspicion": 0.508, "pivotal_queries": 1, "dissimilar_pivots": 0,
  "novelty": 0.86, "verdict": "suspicious",
  "flipped": [{"query": "unknown", "old": "fix_dns", "new": "free_disk",
               "cosine_to_key": 0.492}],
  "cumulative": {"query_corpus": 5, "suspicious_writes": 1,
                 "suspicion_sum": 0.508, "verdict": "clean"}
}}
```

`reject_suspicious: true` returns HTTP 409 and does not write when the verdict
is `suspicious` or the recent cumulative count is at least 3. The check is a
model-sensitivity signal, not proof of malicious intent.

---

Malformed JSON, missing required string fields and invalid limits return
HTTP 400. A loaded AMEM reader has no observations, so SIMILAR, ESCALATE,
NOVELTY, INFLUENCE and COMPACT return HTTP 409. An empty live service returns
no candidates, novelty 1, and an escalation decision. The novelty inverse is
cached until observations change; the first query after a change still builds
it.

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
  "persistence": "memory",
  "labels": 10,
  "d_tfidf": 2127,
  "d_static": 0,
  "solver": "dual",
  "semantic_updates": true,
  "semantic_reads": true,
  "exact_endpoints": ["GET", "PUT", "DELETE", "EXISTS", "SCAN"],
  "semantic_endpoints": ["SIMILAR", "ESCALATE", "NOVELTY"]
}
```

---

## Durability and consistency

The service defaults to **in-memory** operation. With `--data` and
`--encryption-key-file`, successful mutations commit an encrypted, fsynced
snapshot before acknowledgement. Records and absolute TTLs survive restart;
the semantic index is rebuilt. See [durability](docs/DURABILITY.md) for setup,
performance costs and failure semantics. Replication is not implemented.
AMEM model files still preserve classification weights, not records or writer state.

Within a single process:

* Reads are safe concurrently.
* Writes are serialised by one lock, so a `PUT` is atomic with respect to
  other operations.
* The record store is the source of truth for exact operations. The model
  is a secondary index over it.

That last point is the important one: if the model cannot accept a write —
for example a model loaded from disk, whose writer state was dropped — the
exact record store still holds the value, and `GET` still returns it. The
response reports `semantic_updated: false`; the model does not catch up
automatically. Restart with `--train` and the original corpus to build a
live semantic index. Unexpected live-writer failures are returned as errors
before publishing the new record value.

## Deployment

The service uses Kemal 1.15.0. Static-file serving is disabled, framework
errors use the same JSON error envelope, and GET routes support HEAD.
SIGTERM/SIGINT stops accepting new connections and lets Kemal drain
in-flight requests before exiting. The memory engine itself has no Kemal
dependency.

```bash
semrember-server --train corpus.csv --key message --value action
semrember-server --model brain.amem --static models/potion-base-8M.bin
semrember-server --host 0.0.0.0 --port 7379
```

**There is no authentication.** The service binds to `127.0.0.1` by default.
Put it behind something that authenticates before exposing it beyond the
host — it has no notion of tenants or namespaces.
