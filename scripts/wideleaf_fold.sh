#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold side of one wide-leaf chain: 6-fold stack with xtd (scripts/wideleaf_stk.sh) -> C4 / tree D -> clock stages (adsb6 + adsbL clocks, A0 + L) -> scratchpad/xau1/clockfold1_xb<C>_*.
# usage: bash scripts/wideleaf_fold.sh Kq|NNA3        needs data/oof/wideleaf_retune_a2_<fold>_* on all 6 folds (build machine fits)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; C=$1
for f in months39 months28 months410 months511 months612; do ls data/oof/wideleaf_retune_a2_${f}_*.npy >/dev/null; done; ls data/oof/wideleaf_retune_a2_janjul_*.npy >/dev/null
NT=${NT:-4} bash scripts/wideleaf_stk.sh XB$C
clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=${NT:-4} bash scripts/clockstage_fold.sh scratchpad/xbase/P13_XB${C}xTD XB$C
