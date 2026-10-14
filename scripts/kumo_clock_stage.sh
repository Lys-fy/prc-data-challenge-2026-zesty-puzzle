#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold side of the clock stages of the KSTK_CLOCK chain variant: scripts/clockstage_fold.sh with the stage runner scripts/kumo_clock_run.py
# (Kumo member visible to the clock stage; env chain_fold_line_km / chain_fold_line_kmp).
# Outputs under scratchpad/xau1/clockfold1_<name>_* and scratchpad/xlv/leave_stage_xau1_<name>_*; the input TD vectors are only read.
# usage: clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=4 bash scripts/kumo_clock_stage.sh <TD prefix> <out name>
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/xau1; mkdir -p $D; TD=$1; O=$D/clockfold1_$2; T=clockfold1_$2; export oof_build_folds=6; NT=${NT:-4}
export pushstart_adsb6=${pushstart_adsb6:-1} clockfold1_adsbl=${clockfold1_adsbl:-1}
XA=""; [ "${A1:-0}" = 1 ] && XA="--excl-lirf" && export audited_stack_a1=1
if [ "$FORM" = PC ]; then export adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600; else export adsb5_clock_adsb5=0; fi
R=scripts/kumo_clock_run.py   # CLOCK_CHAIN_FIX fix 2: KM visible to the clock stages (dd_kmx_taxi); needs env chain_fold_line_km / chain_fold_line_kmp
$P $R scripts/chain_clock.py $TD $O --ns 6 --nt $NT --objective huber --jit 25 --ncopy 2 --npz ${O}_HuJ.npz $XA | tail -1
leave_stage_noa1=1 $P $R scripts/chain_leave.py ${O}_HuJ.npz --hnpz ${O}_HuJ.npz --tag $T --ns 3 --nt $NT | grep -E "^\s+A0\+L " | cut -c1-120
$P scripts/stage12_a0l.py ${O}_HuJ.npz $T ${O}_HA0L >/dev/null; ls ${O}_HA0L_*.npy | wc -l
