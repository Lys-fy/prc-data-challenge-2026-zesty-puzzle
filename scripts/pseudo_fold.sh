#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold side of one pseudo-label chain (pru=1 exported): 6-fold stack with OL = pseudolabel_<ARM> (scripts/pseudo_stk.sh) -> C4 / tree D ->
# Huber clock stage (adsb6 + adsbL clocks) -> A0 + L -> scratchpad/xau1/clockfold1_xs<ARM><C>_*.     usage: bash scripts/pseudo_fold.sh PL3|B0 Kq|NNA3
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; ARM=$1; C=$2
for f in months39 months28 months410 months511 months612 janjul; do ls data/oof/pseudolabel_${ARM}_${f}_*.npy >/dev/null; done
pru=1 NT=${NT:-4} bash scripts/pseudo_stk.sh $ARM $C
ls scratchpad/xsyn/P13_XS${ARM}${C}xTD_janjul.npy scratchpad/xsyn/P13_XS${ARM}${C}xTDj2_janjul.npy >/dev/null
clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=${NT:-4} bash scripts/clockstage_fold.sh scratchpad/xsyn/P13_XS${ARM}${C}xTD XS${ARM}${C}
ls scratchpad/xau1/clockfold1_xs${ARM}${C}_HA0L_*.npy | wc -l
