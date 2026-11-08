# Installation

[README](../README.md) · [Next: five-minute quickstart](QUICKSTART.md)

## Prerequisites

The source build is exercised on Linux in CI. The commands below use a POSIX
shell; native Windows installation is not covered by this guide.

| Tool | Required for |
|---|---|
| Crystal >= 1.12 and its C toolchain/libraries | Building the executables or using the library |
| Shards | Installing the Kemal service's locked build dependencies |
| Git and Make | Checking out and building this repository |
| curl | The HTTP walkthrough |
| Python >= 3.10 | Python client and test tools; optional for running the service |
| Node.js >= 18 | JavaScript example; optional for other interfaces |

Use the [official Crystal installation instructions](https://crystal-lang.org/install/)
for your operating system. Confirm that both tools are on PATH:

```bash
crystal --version
shards --version
```

## Install the executables

```bash
git clone https://github.com/sethuiyer/semrember.git
cd semrember
make install
export PATH="$HOME/.local/bin:$PATH"
semrember --help
semrember-server --help
```

`make install` builds release executables, installs Kemal 1.15.0 and the
transitive versions in `shard.lock` at build time, and copies both executables
to `~/.local/bin`. Keep the checkout for the example corpus and scripts.
Add the PATH export to your shell configuration if you want it in new shells.

To choose another prefix or stage files for packaging:

```bash
make install PREFIX="$HOME/tools/semrember"
make install PREFIX=/usr DESTDIR="$PWD/build/package"
```

The first installs under `~/tools/semrember/bin`. The second stages files under
`build/package/usr/bin`; it does not write into `/usr/bin`.

To build and run directly from the checkout instead:

```bash
make all
build/semrember-server --train examples/incidents.csv
```

For the CLI alone, `make cli` does not require installing Kemal. The memory
engine can also be imported directly from `src/semrember.cr`.

## Install the Python client

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install ./clients/python
.venv/bin/python -c 'from semrember_client import Semrember; print(Semrember)'
```

Use `.venv/bin/python` for the examples, or activate the virtual environment.
The distribution is named `semrember-client`; the import is `semrember_client`.
This command installs from the local checkout, not from a published PyPI release.
The client has no third-party runtime dependencies and connects to a separately
running service. It does not install or start the server.

For client development, use `pip install -e ./clients/python` in your environment.
For a source-only import without installation, set `PYTHONPATH=clients/python`.

## Optional static embeddings

The default mode needs no embedding download, GPU, or LLM API key. To add a
static encoder after trying the basic walkthrough:

```bash
make models-base
build/semrember-server --train examples/incidents.csv \
  --static models/potion-base-8M.bin
```

Model fetching requires network access. The encoder consumes additional disk
and resident memory. See [model sources and formats](../MODELS.md). When saving
a hybrid AMEM model, retain the original encoder file separately.

## Binary portability

The default build links system libraries. Installing Kemal during the build
does not mean a separate Kemal process or package is required at runtime.
Do not assume a default binary can be copied to any operating system.

Static linking is optional on a toolchain with the required static libraries:

```bash
make CRYSTAL_FLAGS="--release --static"
```

## Troubleshooting

| Symptom | What to check |
|---|---|
| `crystal` or `shards` not found | Install the missing tool and put its executable on PATH |
| Compiler/linker reports a missing library | Install the development libraries required by your Crystal distribution |
| `semrember-server` not found after install | Export the selected prefix's `bin` directory on PATH, or run `build/semrember-server` |
| `Address already in use` | Stop the existing service or use `--port 7380`; use the same port in clients |
| Python cannot import `semrember_client` | Install with the same interpreter used to run the script; use the virtual environment's Python |
| Semantic endpoint returns HTTP 409 | Start with `--train`, not an AMEM reader loaded with `--model` |

To update, pull the source and rerun `make install`. To remove the default
installation, remove only `~/.local/bin/semrember` and
`~/.local/bin/semrember-server`. Application records and model files are separate.
