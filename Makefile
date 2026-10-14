# zesty-puzzle, PRC Data Challenge 2026: entry points (README.md, docs/REPRODUCE.md).
# A = the one-off 2025 side (member OOF, chain fold sides, 2025 constants); B = the production run on a ranking file.
# Run from the repository root: scripts take the root from PRC_ROOT (default: the current directory), from their own location (the parent of scripts/) or as paths relative to the current directory.

SHELL   := /bin/bash
PY      ?= .venv/bin/python
TORCH_PY ?= $(CURDIR)/.venv-torch/bin/python
KUMO_PY  ?= $(CURDIR)/.venv-kumo/bin/python
PERIOD  ?= final
PRU_DIR ?= data/external/pru_20261008
OPDI_PERIODS ?= 2025-01-01:2025-12-31,2026-01-01:2026-02-28,2026-05-31:2026-07-31
export PRC_ROOT := $(CURDIR)
export PYTHONHASHSEED := 0
export TORCH_PY KUMO_PY PRU_DIR

.PHONY: help env data external adsb-extract features folds meta-oof constants final assemble validate baseline test check-release clean-work

help:
	@echo "A (once):  env data external adsb-extract features meta-oof folds constants"
	@echo "B:         final [STAGE=...] validate   PERIOD=$(PERIOD)  (phase2/periods/<PERIOD>.env); assemble STAGES=<dir>"
	@echo "baseline:  baseline    (laptop baseline, baseline/README.md)"
	@echo "release:   test check-release"

env:            ## main env (uv.lock) + checks of the torch and Kumo envs (env/README.md)
	uv sync
	$(PY) -c "import lightgbm, catboost, polars; print('main env ok', lightgbm.__version__, catboost.__version__, polars.__version__)"
	$(TORCH_PY) -c "import torch, pytabkit; print('torch env ok', torch.__version__)" || echo "torch env missing: see env/README.md"
	$(KUMO_PY) -c "import sdm, torch; print('kumo env ok', torch.__version__)" || echo "kumo env missing: see env/README.md"

data:           ## competition files with your own team credentials (not distributed)
	$(PY) scripts/download_data.py --bucket prc-2026-datasets
	@ls data/raw/training_*.parquet >/dev/null && test -f data/raw/ranking.parquet

external: data  ## open data (docs/DATA_SOURCES.md): METAR, OPDI (~15 GB), PRU csv, OpenFlights
	$(PY) scripts/download_metar.py && $(PY) scripts/build_metar.py
	$(PY) scripts/fetch_opdi.py --periods $(OPDI_PERIODS)
	$(PY) scripts/build_opdi_features.py
	$(PY) scripts/fetch_pru.py --out $(PRU_DIR)
	$(PY) scripts/pru_build.py
	mkdir -p data/external/openflights && curl -fsSL -o data/external/openflights/airlines.dat \
	  https://raw.githubusercontent.com/jpatokal/openflights/master/data/airlines.dat

adsb-extract: data   ## adsb.lol archives -> airport-box points -> per-departure clock day files (resumable)
	STANDPOS_DAYS=manifest/standpos_days.txt PY=$(PY) bash scripts/adsb_all.sh pull extract standpos extract2 extract3 extract5 extract6 extractL

features: external   ## A: features_v9 over 2025 + the public ranking file + per-row side tables (+ the ADS-B side tables)
	$(PY) scripts/build_features.py --out data/features_v9.parquet
	$(PY) scripts/build_xba_features.py && $(PY) scripts/xba_fallback_ids.py
	$(PY) scripts/build_xe_queue_features.py
	$(PY) scripts/build_dreg_features.py
	$(PY) scripts/build_durp_features.py
	$(PY) scripts/queue_regime_prob.py prod
	$(PY) scripts/sealed_days.py
	PY=$(PY) bash scripts/adsb_all.sh tables

folds:          ## A1 + A2: member OOF and chain fold sides (scripts/a_side.sh; DRY=1 prints the commands); then make constants
	bash scripts/a_side.sh

meta-oof:       ## A1: fold OOF of the chain meta inputs (before the chain fold sides of A2)
	bash scripts/constants_2025.sh meta

constants:      ## A3: refit the 2025 constants from the A-side fold vectors (pool: env POOL_CHAINS, default the vectors of the shipped fit)
	bash scripts/constants_2025.sh pool postrules tw1 krs krsnl exposed

final:          ## B: production run by stage (STAGE = inputs members kumo line kstk levers bag assemble; default all) -> phase2/work/$(PERIOD)/scratchpad/final_assembly/zesty-puzzle_final.parquet
	bash scripts/run_final.sh $(PERIOD) $(STAGE)

assemble:       ## B, last stage alone: the submitted file from the two saved line outputs (STAGES = dir of manifest/final_inputs.md5), docs/REPRODUCE.md §6
	STAGES=$(STAGES) bash scripts/assemble_final.sh $(or $(OUT),out/final)

validate:       ## schema, ids, >= 60 s floor, LOBT bound
	$(PY) scripts/validate_submission.py phase2/work/$(PERIOD)/scratchpad/final_assembly/zesty-puzzle_final.parquet --features phase2/work/$(PERIOD)/data/features_v9.parquet

baseline:       ## laptop baseline: 2025 fold RMSEs, then a submission for the final ranking file (outputs in data/baseline/)
	$(PY) -m baseline.run evaluate --raw-dir data/raw --out data/baseline
	$(PY) -m baseline.run predict --raw-dir data/raw --ranking data/final/final_ranking.parquet --out data/baseline

test:           ## ruff + unit tests on synthetic rows (tests/, baseline/tests/; no data) as in .github/workflows/ci.yml; dev tools from the uv dev group (uv sync --group dev)
	$(PY) -m ruff check . && $(PY) -m pytest -q

check-release:  ## pre-publication scan: no data files, secrets, credentials or host paths
	bash tools/check_release.sh .

clean-work:     ## remove pipeline workspaces (never touches data/)
	rm -rf phase2/work/*
