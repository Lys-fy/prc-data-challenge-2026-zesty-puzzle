#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# BIG_NOVELTY chain variant: the k250_stack chains (KM=KSBWPc100) with an airport-day set-transformer member as xtd3 (env big, default big_ctx3), Kq and NNA3, fold side then production side. CPU only.
# usage: NT=8 bash scripts/dayset_chain.sh fold|prod Kq|NNA3     (after: python scripts/dayset_prep.py)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; W=$1; C=$2; export NT=${NT:-8} KM=KSBWPc100 KMP=KSBWPc100P PFX=${PFX:-big} big=${big:-big_ctx3} extra3_mask=${extra3_mask:-} BIG_PROD_PREFIX=${BIG_PROD_PREFIX:-big_ctxp} NXDS=3   # BIG_MEMBER_OVERRIDE: member overridable (big_big3 / big_bigp)
echo "big chain $W $C PFX=$PFX MASK=$extra3_mask NT=$NT $(hostname) $(date -u +%FT%TZ)"; md5sum scripts/dayset_stack.py scripts/dayset_assemble.py data/oof/${big}_janjul.npy | cut -c1-12,34-
case $W in
  fold) bash scripts/dayset_fold.sh $C;;
  prod) pru=1 FORM=PC A1=1 FBMODE=member ARM=KUMO_CLOCKFOLD_ARM ARMIN=XK SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1 bash scripts/dayset_bundle_prod.sh $C;;
  *) echo "W?"; exit 1;;
esac; echo "big chain done $W $C $(date -u +%FT%TZ)"
