# Five-minute quickstart

[Installation](INSTALLATION.md) · [API examples](EXAMPLES.md) · [Memory and persistence](MEMORY.md)

This walkthrough starts after Crystal, Shards, and the executables are
installed. A first compilation or optional model download can take longer.
Run commands from the cloned repository root. Use a fresh example service;
the overwrite below changes one in-memory record until you stop it.

## 1. Start the service

In terminal A:

```bash
build/semrember-server --train examples/incidents.csv --key message --value action
```

The included CSV contains text in `message` and a suggested action in `action`.
Kemal listens at `http://127.0.0.1:7379`. Leave this terminal running.
No static model or API key is needed.

## 2. Retrieve an exact record

In terminal B:

```bash
curl -fsS http://127.0.0.1:7379/v1/ping
curl -fsS http://127.0.0.1:7379/v1/info
curl -fsS http://127.0.0.1:7379/v1/kv/certificate%20has%20expired
```

The last response is:

```json
{"key":"certificate has expired","value":"renew_certificate"}
```

Exact GET uses the record dictionary and a case-sensitive key. It does not
ask the model to predict a value. Encode the whole key once as a URL path
segment, including `/` as `%2F` and literal `+` as `%2B`.

## 3. Find related examples and ask for a decision

```bash
curl -fsS http://127.0.0.1:7379/v1/similar \
  -H 'Content-Type: application/json' \
  -d '{"text":"certificate has expired","top_k":2}'

curl -fsS http://127.0.0.1:7379/v1/escalate \
  -H 'Content-Type: application/json' \
  -d '{"text":"certificate has expired"}'
```

SIMILAR returns ranked records with `key`, `value`, and `score`. ESCALATE should
return `decision: "handle"`, `handled: true`, and `action: "renew_certificate"`
for this training example. Its response also includes candidates, margin,
novelty, and a reason. Numeric scores can vary with training data and precision.

The decision is a suggestion. Your application decides whether to perform an
action. For an unfamiliar or ambiguous input, inspect `handled` and `reason`;
when escalating, `action` is null. Scores are not calibrated probabilities.

## 4. Replace a stored value

```bash
curl -fsS -X PUT http://127.0.0.1:7379/v1/kv/certificate%20has%20expired \
  -H 'Content-Type: application/json' \
  -d '{"value":"inspect_certificate"}'

curl -fsS http://127.0.0.1:7379/v1/kv/certificate%20has%20expired
```

The write reports `previous: "renew_certificate"` and `semantic_updated: true`.
GET now returns `inspect_certificate`. The live dual writer also changes the
training target for that exact key. Updating the target is exact algebra for
the ridge objective; predictions for arbitrary inputs still require evaluation.

Restore the example before running the scripts below:

```bash
curl -fsS -X PUT http://127.0.0.1:7379/v1/kv/certificate%20has%20expired \
  -H 'Content-Type: application/json' \
  -d '{"value":"renew_certificate"}'
```

## 5. Connect an application

With the Python client installed as described in the installation guide:

```bash
.venv/bin/python examples/quickstart.py
```

It runs exact reads, similarity, escalation, replacement, scan, and TTL,
then removes its temporary demo key. The bundled training records are retained.

JavaScript users can run the dependency-free Node.js example:

```bash
node examples/quickstart.mjs
```

Crystal users can run the engine directly, with no service:

```bash
crystal run examples/quickstart.cr
```

The last script demonstrates a learned label, replacement, saving, loading,
and the read-only nature of an AMEM reader.

## Stop, restart, and bring your own data

Stop terminal A with Ctrl+C. Kemal drains current requests before exiting.
Restarting with the same CSV rebuilds the original records; it does not restore
changes made through PUT or their TTLs. This walkthrough never edits the CSV.
To retain live writes and TTLs instead, start with [encrypted durable storage](DURABILITY.md).

For your own corpus, use a UTF-8 CSV with a header and string keys/labels:

```csv
message,action
card declined,review_payment
payment rejected,review_payment
password reset requested,reset_password
```

```bash
build/semrember-server --train your-data.csv --key message --value action
```

An exact duplicate key uses its last value when loading the CSV. Choose
representative examples up front: the vocabulary is fixed at fit time.
See [memory and persistence](MEMORY.md) before relying on live writes.
