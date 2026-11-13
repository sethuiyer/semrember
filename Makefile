CRYSTAL ?= crystal
SHARDS ?= shards
CRYSTAL_FLAGS ?= --release
BUILD   ?= build
PREFIX  ?= $(HOME)/.local
DESTDIR ?=
DX_PYTHON ?= python3

.PHONY: all deps cli server spec test parity bench models models-base models-multilingual clean fetch-model smoke install dx-smoke pilot-insurance pilot-insurance-live pilot-test bench-writer verify-sparse-x influence-test banzhaf-mc bench-banzhaf pokemon-banzhaf retention poison-check compact-pilot maxsat shard-plan

all: cli server

install: all
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 755 "$(BUILD)/semrember" "$(BUILD)/semrember-server" "$(DESTDIR)$(PREFIX)/bin/"

$(BUILD):
	mkdir -p $(BUILD)

cli: | $(BUILD)
	$(CRYSTAL) build $(CRYSTAL_FLAGS) src/cli.cr -o $(BUILD)/semrember

# Standalone weighted MaxSAT approximator (the NuWLS core in Crystal).
maxsat: | $(BUILD)
	$(CRYSTAL) build $(CRYSTAL_FLAGS) src/maxsat_cli.cr -o $(BUILD)/maxsat

deps: lib/.semrember-deps

lib/.semrember-deps: shard.yml shard.lock
	$(SHARDS) install --frozen
	touch $@

server: deps | $(BUILD)
	$(CRYSTAL) build $(CRYSTAL_FLAGS) src/server_cli.cr -o $(BUILD)/semrember-server

spec: deps
	$(CRYSTAL) spec

# Maintainer tool.  Compares this implementation against the reference
# implementations, which are NOT part of this repository — see the note in
# tools/parity.py.  A public checkout cannot run this; `make spec` is the
# test suite that ships.
#   make parity
#   python3 tools/parity.py --static models/potion-base-8M.bin
parity: cli
	python3 tools/parity.py $(PARITY_ARGS)

# `test` is what ships: no reference implementation required.
test: spec
	python3 -m unittest discover -s clients/python -p 'test_*.py'
	python3 examples/pilot_insurance_ca/test_policy.py

smoke:
	python3 tools/http_smoke.py --server $(BUILD)/semrember-server --cli $(BUILD)/semrember

.PHONY: durable-smoke
durable-smoke:
	python3 tools/durable_smoke.py --server $(BUILD)/semrember-server

dx-smoke:
	python3 tools/dx_smoke.py --server $(BUILD)/semrember-server --cli $(BUILD)/semrember --python "$(DX_PYTHON)" --crystal "$(CRYSTAL)"

# Canadian insurance FNOL triage pilot. Offline by default; the live target
# calls TypeSafe System One and requires TYPESAFE_API_KEY or --key-file.
pilot-insurance:
	$(DX_PYTHON) examples/pilot_insurance_ca/demo.py

pilot-insurance-live:
	$(DX_PYTHON) examples/pilot_insurance_ca/demo.py --live

pilot-test:
	$(DX_PYTHON) examples/pilot_insurance_ca/test_policy.py

bench: cli
	AMEM=$(AMEM) $(CRYSTAL) run --release tools/bench.cr

bench-writer:
	$(CRYSTAL) run --release tools/bench_writer.cr

SPARSE_N ?= 1000
verify-sparse-x:
	$(CRYSTAL) run --release tools/verify_sparse_x.cr -- $(SPARSE_N)

influence-test:
	$(CRYSTAL) spec spec/influence_spec.cr

banzhaf-mc:
	$(CRYSTAL) run --release tools/banzhaf_mc.cr

BANZHAF_N ?= 1000
bench-banzhaf:
	$(CRYSTAL) run --release -Dlapack tools/bench_banzhaf.cr -- $(BANZHAF_N)

POKEMON_CSV ?= $(HOME)/semrember/build/pokemon.csv
pokemon-banzhaf:
	$(CRYSTAL) run --release examples/pokemon_banzhaf.cr -- $(POKEMON_CSV)

retention:
	$(CRYSTAL) run --release tools/retention_experiment.cr -- $(POKEMON_CSV)

poison-check:
	$(CRYSTAL) run --release examples/poison_check.cr

COMPACT_CSV ?= $(HOME)/Documents/Documents/Workspace/Datasets/BBC_News_Article_Train.csv
COMPACT_TEXT ?= text
COMPACT_LABEL ?= category
COMPACT_ROWS ?= 1200
compact-pilot:
	$(CRYSTAL) run --release tools/compact_experiment.cr -- $(COMPACT_CSV) --text-col $(COMPACT_TEXT) --label-col $(COMPACT_LABEL) --max-rows $(COMPACT_ROWS) --queries 100 --coverage-floor 2

# Static embedding models.  Not redistributed: fetched from the Hugging Face
# hub and converted to the M2VE/M2VS format, verified by checksum.  No-op if
# the file is already present.
MODELS_DIR ?= models

fetch-model: | $(BUILD)
	$(CRYSTAL) build $(CRYSTAL_FLAGS) tools/fetch_model.cr -o $(BUILD)/fetch-model

models: models-multilingual

models-base: fetch-model
	$(BUILD)/fetch-model base --out $(MODELS_DIR)

models-multilingual: fetch-model
	$(BUILD)/fetch-model multilingual --out $(MODELS_DIR)

clean:
	rm -rf $(BUILD)

.PHONY: mips-test bench-mips
mips-test:
	$(CRYSTAL) spec spec/mips_spec.cr

bench-mips:
	$(CRYSTAL) run $(CRYSTAL_FLAGS) tools/bench_mips.cr -- $(MIPS_ARGS)

SHARD_ARGS ?= examples/incidents.csv --key message --value action --budget 20
shard-plan:
	$(CRYSTAL) run $(CRYSTAL_FLAGS) tools/shard_plan.cr -- $(SHARD_ARGS)
