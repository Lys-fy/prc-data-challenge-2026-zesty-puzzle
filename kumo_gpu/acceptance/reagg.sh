#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# RERUN_PATCH 3 Oct: re-run the agg units that failed in the night run (km_engine.py agg indexed the airport frame with global _row ids; fixed 3 Oct).
# CPU-side only (no GPU work); safe next to running inst workers. From the bundle root: bash acceptance/reagg.sh
set -uo pipefail; B=$(cd "$(dirname "$0")/.." && pwd); cd "$B"
export BUNDLE=$B PAYLOAD=${PAYLOAD:-$B/data/payload} RUN=${RUN:-$B/run} ENVDIR=${ENVDIR:-$B/.venv-kumo}; export HF_HOME=${HF_HOME:-$PAYLOAD/vendor/hf} HF_HUB_OFFLINE=1 PRC_THREADS=${PRC_THREADS:-4}
PY=$ENVDIR/bin/python; APTS="LTFM EHAM EGLL LFPG EDDF LEMD LEBL EDDM LIRF LSZH"
spec_of() { grep "^$1 " acceptance/$2 | sed 's/^[^ ]* *//; s/ \+[0-9]\+ .*$//' | tr ' ' '\n' | grep -v '^FOLDS=\|^APTS=' | tr '\n' ' '; }
ag() { local NAME=$1 MOD=$2 FOLDS=$3 AP=$4; shift 4; local G=""; for f in ${FOLDS//,/ }; do for a in $AP; do
    [ -f "$RUN/results/member/${NAME}_${f}_$a.npz" ] && continue; [ -d "$RUN/results/engine/$NAME/${f}_$a" ] && G="$G agg:$f:$a"; done; done
  [ -z "$G" ] && return 0; env "$@" NAME=$NAME CUDA_VISIBLE_DEVICES= DEV=cpu $PY $MOD $G 2>&1 | grep -E "agg\[|Error|assert" | sed "s/^/  /"; }
for NM in KSB60c100 KSB120c100 KSBWPc100; do ag $NM cluster/km_engine.py janjul,months28,months39,months410,prod "$APTS" $(spec_of $NM sb_profiles.txt); done
for NM in KSBWPc100cl60 KME8r0 KME8pi; do ag $NM cluster/km_engine.py months28 "LTFM EHAM LSZH" $(spec_of $NM r0_profiles.txt); done
for NM in KME8c100 KLE8c100 KME8c100m KME16c100 KME8c100r KME4c100x50 KME32c100; do ag $NM cluster/km_engine.py janjul,months28,months39,months410,prod "$APTS" $(spec_of $NM session1_profiles.txt); done
RESID_DIR=$B/data/resid ag kumo_resid_adapter_r1 cluster/km_resid.py janjul,months28 "$APTS" RESID_DIR=$B/data/resid $(spec_of kumo_resid_adapter_r1 resid_profiles.txt)
for V in "ABL50 50 0 " "ABL100 100 0 " "ABL412 412 0 " "ABL100n20 100 20 " "ABL100d20 100 0 EXTRACOLS=$B/data/km_drift20.txt"; do set -- $V
  ag $1 cluster/km_engine.py janjul "LSZH EDDM" SIZE=medium CAP=20000 NCOL=$2 NOISE=$3 COLMODE=imp CORE=50 NINST=1 EIN=4 ${4:-}; done
echo REAGG_DONE
