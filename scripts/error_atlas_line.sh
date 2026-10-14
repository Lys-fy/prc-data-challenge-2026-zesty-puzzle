#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Matched 2025 fold line of the error table (v94 fold equivalent) and the table itself.
# (B) + (A) on three chain sets (LdTK + the Kq / NNA3 chains of the arm), w5 pool .3212/.3065/.3723, unmatched day sums from um25_A1:
#   XB      -> scratchpad/xbase/wideleaf_le3{B,BA}_<fold>.npy      (bash scripts/wideleaf_fold.sh Kq|NNA3)
#   XS PL3  -> scratchpad/xsyn/pseudolabel_pl3_le3{B,BA}_<fold>.npy    (bash scripts/pseudo_fold.sh PL3 Kq|NNA3)
#   XS B0   -> scratchpad/xsyn/pseudolabel_b0_le3{B,BA}_<fold>.npy     (bash scripts/pseudo_fold.sh B0 Kq|NNA3; members: pseudo_member.sh B0 <fold>)
# then scripts/error_atlas_tables.py -> scratchpad/xatlas/rows_matched.parquet.
# usage: bash scripts/error_atlas_line.sh
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=${PY:-.venv/bin/python}; A=scratchpad/xau1; W5=0.3212,0.3065,0.3723; UM=scratchpad/xau3/um25_A1.npz
mkdir -p scratchpad/xbase scratchpad/xsyn
line(){   # line(out prefix, chain tag)
  local CH="$A/clockfold1_ldtk_ha0l,$A/clockfold1_$2Kq_HA0L,$A/clockfold1_$2NNA3_HA0L"
  ls $A/clockfold1_ldtk_ha0l_janjul.npy $A/clockfold1_$2Kq_HA0L_janjul.npy $A/clockfold1_$2NNA3_HA0L_janjul.npy >/dev/null
  $P scripts/pru_daily_calib.py --pool $CH --w $W5 --out $1 --um25 $UM > $1.log 2>&1
  grep -E "INFO B " $1.log | head -2 || true
}
line scratchpad/xbase/wideleaf_le3 XB
line scratchpad/xsyn/pseudolabel_pl3_le3 PSEUDOLABEL_CHAIN
line scratchpad/xsyn/pseudolabel_b0_le3 PSEUDO_B0_LINE
$P scripts/error_atlas_tables.py --out scratchpad/xatlas
