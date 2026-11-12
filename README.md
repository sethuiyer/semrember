# semrember

Semrember learns **text → label** associations and serves them through a local
HTTP API. It combines byte n-gram TF-IDF, optional static embeddings, and a
ridge-regression writer. Use it to suggest an incident action, classify an
intent, or retrieve related examples from your own corpus.

It provides three different behaviors:

| Operation | What you get |
|---|---|
| Exact `GET` | The stored string for exactly that key |
| `SIMILAR` | Related stored keys, their values, and similarity scores |
| `ESCALATE` | A suggested label or an escalation decision, with scores and reasons |

A learned prediction can be wrong. Scores are not probabilities. Semrember
does not generate answers or reconstruct arbitrary documents from its weights.

## Install

Requires Crystal >= 1.12, Shards, Git, Make, and the system C toolchain/libraries.
Building the HTTP service also requires libsodium development headers
(`libsodium-dev` on Debian/Ubuntu); the engine and standalone CLI do not.
Follow the [Crystal installation guide](https://crystal-lang.org/install/) for
your platform, then run:

```bash
git clone https://github.com/sethuiyer/semrember.git
cd semrember
make install
export PATH="$HOME/.local/bin:$PATH"
```

This builds the CLI and Kemal service, and installs them to `~/.local/bin`.
Use `make install PREFIX=/your/prefix` to choose another destination.
No model download is needed for the default lexical mode.

See [installation and troubleshooting](docs/INSTALLATION.md) for prerequisites,
build-only use, the Python client, and optional embeddings.

## Five-minute quickstart

After installation, start a live service from the included examples:

```bash
semrember-server --train examples/incidents.csv --key message --value action
```

In another terminal:

```bash
curl -fsS http://127.0.0.1:7379/v1/ping

curl -fsS http://127.0.0.1:7379/v1/escalate \
  -H 'Content-Type: application/json' \
  -d '{"text":"certificate has expired"}'
```

The second response should have `handled: true` and
`action: "renew_certificate"`. No LLM API key is required.

Follow the [full five-minute walkthrough](docs/QUICKSTART.md) to try exact
reads, similarity, an overwrite, and the Python client. Initial toolchain
installation and compilation can take longer than the walkthrough.

## Use from your application

| Interface | Start here |
|---|---|
| HTTP / curl | [API reference](API.md) and [API examples](docs/EXAMPLES.md) |
| Python 3.10+ | `python -m pip install ./clients/python`, then [runnable example](examples/quickstart.py) |
| JavaScript (Node.js >= 18) | [Runnable example](examples/quickstart.mjs), using built-in `fetch` |
| Crystal library | [Runnable example](examples/quickstart.cr), with no HTTP or Kemal dependency |
| Offline CLI | [Save and query a model](docs/MEMORY.md#save-a-classification-reader) |

The Python install command uses this checkout. There is no hosted-package
installation or prebuilt-binary release assumed by these instructions.

## Pilot demo: Canadian insurance FNOL triage

[`examples/pilot_insurance_ca/`](examples/pilot_insurance_ca/README.md) is a
runnable first pilot for a fictional Canadian P&C insurer. Semrember retrieves
similar claims, past operational scars, fairness lessons, and attack patterns;
Jev / System One supplies typed judgment; code routes to a human-owned queue.

```bash
make pilot-insurance        # offline, local-only fallback
make pilot-insurance-live   # requires a TypeSafe API key
```

The demo uses synthetic data, never denies or pays a claim, and writes an
optional HTML walkthrough and JSON audit. It is a routing demonstration, not a
production claims decision system.

## Understand the memory before integrating

**The live record dictionary and a saved model are different things.**

- A running `--train` service keeps exact records and live observations.
- An AMEM file saves vocabulary, IDF, labels, and float16 prediction weights.
- AMEM does **not** save record keys, TTLs, training observations, or writer state.
- Default mode loses live writes on restart. Opt into [encrypted durable storage](docs/DURABILITY.md)
  with `--data state.enc --encryption-key-file secret.key` to recover records and TTLs.
- A `--model` service starts with an empty exact store. Its writes report
  `semantic_updated: false`; similarity, novelty, and escalation return HTTP 409.
- In default mode, vocabulary is fixed after fitting. New keys can update weights only through
  existing features. Refit from a representative corpus to expand vocabulary.

Read [what it can remember and how persistence works](docs/MEMORY.md) before
using it as a cache, router, or long-running service. Durable mode restores exact
records and rebuilds a writable semantic index on restart.

## Develop and verify

```bash
make all
make test
make smoke
```

For the executable documentation checks, also install the Python client and
Node.js, then follow [the example-check instructions](docs/EXAMPLES.md#verify-the-examples).
CI checks the examples, installation target, and persistence behavior.

## Further reading

- [Installation](docs/INSTALLATION.md)
- [Five-minute quickstart](docs/QUICKSTART.md)
- [API examples](docs/EXAMPLES.md)
- [Memory and persistence](docs/MEMORY.md)
- [HTTP API contract](API.md)
- [Optional static embedding models](MODELS.md)
- [Engine design and historical measurements](docs/DESIGN.md)
- [Decision power and leave-one-out influence](docs/INFLUENCE.md)
- [15-row worked example](docs/WORKED_EXAMPLE.md)
- [Retention pilot study](docs/RETENTION_STUDY.md)
- [AMEM v3 binary format](FORMAT.md)

The service binds to `127.0.0.1:7379` by default and has no authentication.
For deployment details, see [the API deployment notes](API.md#deployment).

MIT license. Optional embedding models have their own terms; see [MODELS.md](MODELS.md).
