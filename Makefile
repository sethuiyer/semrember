CRYSTAL ?= crystal
BUILD   ?= build

.PHONY: all cli server spec test parity bench models models-base models-multilingual clean

all: cli server

$(BUILD):
	mkdir -p $(BUILD)

cli: | $(BUILD)
	$(CRYSTAL) build --release src/cli.cr -o $(BUILD)/semrember

server: | $(BUILD)
	$(CRYSTAL) build --release src/server.cr -o $(BUILD)/semrember-server

spec:
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

bench: cli
	AMEM=$(AMEM) $(CRYSTAL) run --release tools/bench.cr

# Static embedding models.  Not redistributed: fetched from the Hugging Face
# hub and converted to the M2VE/M2VS format, verified by checksum.  No-op if
# the file is already present.
MODELS_DIR ?= models

fetch-model: | $(BUILD)
	$(CRYSTAL) build --release tools/fetch_model.cr -o $(BUILD)/fetch-model

models: models-multilingual

models-base: fetch-model
	$(BUILD)/fetch-model base --out $(MODELS_DIR)

models-multilingual: fetch-model
	$(BUILD)/fetch-model multilingual --out $(MODELS_DIR)

clean:
	rm -rf $(BUILD)
