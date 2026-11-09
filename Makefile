CRYSTAL ?= crystal
SHARDS ?= shards
CRYSTAL_FLAGS ?= --release
BUILD   ?= build
PREFIX  ?= $(HOME)/.local
DESTDIR ?=
DX_PYTHON ?= python3

.PHONY: all deps cli server spec test parity bench models models-base models-multilingual clean fetch-model smoke install dx-smoke

all: cli server

install: all
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 755 "$(BUILD)/semrember" "$(BUILD)/semrember-server" "$(DESTDIR)$(PREFIX)/bin/"

$(BUILD):
	mkdir -p $(BUILD)

cli: | $(BUILD)
	$(CRYSTAL) build $(CRYSTAL_FLAGS) src/cli.cr -o $(BUILD)/semrember

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

smoke:
	python3 tools/http_smoke.py --server $(BUILD)/semrember-server --cli $(BUILD)/semrember

dx-smoke:
	python3 tools/dx_smoke.py --server $(BUILD)/semrember-server --cli $(BUILD)/semrember --python "$(DX_PYTHON)" --crystal "$(CRYSTAL)"

bench: cli
	AMEM=$(AMEM) $(CRYSTAL) run --release tools/bench.cr

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
