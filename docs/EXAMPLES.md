# API examples

[Installation](INSTALLATION.md) · [Quickstart](QUICKSTART.md) · [Full API reference](../API.md)

Start the bundled live service before running HTTP examples:

```bash
build/semrember-server --train examples/incidents.csv
```

## curl

Read an exact string and retrieve similar records:

```bash
curl -fsS http://127.0.0.1:7379/v1/kv/certificate%20has%20expired
curl -fsS http://127.0.0.1:7379/v1/similar \
  -H 'Content-Type: application/json' \
  -d '{"text":"certificate has expired","top_k":3,"min_score":0.1}'
```

Write a key that contains a literal slash, inspect its expiry, then remove it:

```bash
curl -fsS -X PUT http://127.0.0.1:7379/v1/kv/example%2Fitem \
  -H 'Content-Type: application/json' -d '{"value":"review"}'
curl -fsS http://127.0.0.1:7379/v1/kv/example%2Fitem/exists
curl -fsS -X POST http://127.0.0.1:7379/v1/kv/example%2Fitem/expire \
  -H 'Content-Type: application/json' -d '{"seconds":60}'
curl -fsS http://127.0.0.1:7379/v1/kv/example%2Fitem/ttl
curl -fsS -X DELETE http://127.0.0.1:7379/v1/kv/example%2Fitem
```

Encode the key once, then append an operation suffix. `a%2Fexists` is a literal
key named `a/exists`; `a/exists` is the EXISTS operation on the key `a`.

## Python

Install the client from this checkout and run the complete example:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ./clients/python
.venv/bin/python examples/quickstart.py
```

The [example source](../examples/quickstart.py) exercises records, similarity,
decisions, replacement, scan, and TTL. Your application can handle errors by status:

```python
from semrember_client import Semrember, SemremberError

memory = Semrember("http://127.0.0.1:7379")
try:
    result = memory.escalate("certificate has expired")
    if result["handled"]:
        print("Suggested action:", result["action"])
    else:
        print("Escalate:", result["reason"], result["candidates"])
except SemremberError as error:
    print("Request failed:", error.status, str(error))
```

`get()` returns `None` for an absent exact key; other HTTP errors raise
`SemremberError`. The client encodes keys for you. Its `set()` method returns
the previous exact value; inspect `/v1/info` for semantic-update availability.
The raw HTTP PUT response also includes `semantic_updated`.

## JavaScript

```bash
node examples/quickstart.mjs
```

The [example source](../examples/quickstart.mjs) uses Node.js >= 18 with built-in
`fetch`, checks HTTP error statuses, and percent-encodes each complete key.
It needs no npm package. For a different server:

```bash
SEMREMBER_URL=http://127.0.0.1:7380 node examples/quickstart.mjs
.venv/bin/python examples/quickstart.py --url http://127.0.0.1:7380
```

## Crystal library

```bash
crystal run examples/quickstart.cr
```

The [example source](../examples/quickstart.cr) imports `src/semrember.cr`
directly. It fits a live dual writer, replaces a label, saves a temporary
AMEM file, queries the loaded reader, and demonstrates why writes need a
refit after loading. It removes its temporary file on exit.

## Verify the examples

From the checkout, with Crystal, Python, and Node.js on PATH:

```bash
make all test smoke
python3 -m venv .venv
.venv/bin/python -m pip install ./clients/python
make dx-smoke DX_PYTHON="$PWD/.venv/bin/python"
```

The developer-experience check starts its own servers on temporary local ports,
runs the Python and JavaScript examples, executes the Crystal example, stages
an installation, and checks CLI save/load plus HTTP restart behavior. No
optional embedding model or external service is used. CI runs this check too.
