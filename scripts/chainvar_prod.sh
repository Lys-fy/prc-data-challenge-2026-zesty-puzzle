#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Production side of one chain variant (wide-leaf member + KM as extra inputs): pre (meta on ranking rows) -> C4 -> tree D -> Huber clock stage -> A0 + L.
# usage: KM=kumo_member_screen_km KMP=kumo_member_screen_kmp PFX=kumo_member NT=10 bash scripts/chainvar_prod.sh Kq|NNA3
#        needs the fold job (scratchpad/xhook/P13_<PFX><C>x*, scratchpad/xau1/clockfold1_<PFX><C>_HuJ.npz) and the production members (chainvar_check.py inputs prod)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; C=$1; export KM=${KM:?} KMP=${KMP:?} PFX=${PFX:?}
.venv/bin/python scripts/chainvar_check.py inputs prod $C
pru=1 FORM=PC A1=1 FBMODE=member NT=${NT:-4} ARM=KUMO_CLOCKFOLD_ARM ARMIN=XK SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1 bash scripts/chainvar_bundle_prod.sh $C
ls -la scratchpad/xhook/PR_${PFX}${C}_XKAU1_HA0L.parquet
