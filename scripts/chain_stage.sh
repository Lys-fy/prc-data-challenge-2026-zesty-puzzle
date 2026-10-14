#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# Fold side of the clock stages of a chain: clock stage + leave stage (strict nesting) + A0 + L on a 6-fold TD vector.
# FORM=PB (jittered Huber clock stage) | PC (+ adsb5 clocks, 255 leaves, 600 rounds); A1=1: LIRF out of the stage fits (outer and inner stage 1).
# usage: FORM=PC A1=0 bash scripts/chain_stage.sh <TD prefix> <out name>
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; D=scratchpad/p13; TD=$1; O=$D/P13_$2; T=P13_$2; export oof_build_folds=6; NT=${NT:-6}
XA=""; [ "${A1:-0}" = 1 ] && XA="--excl-lirf" && export audited_stack_a1=1
if [ "$FORM" = PC ]; then export adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600; R=scripts/adsb5_stage_run.py; else export adsb5_clock_adsb5=0; R=scripts/adsb5_stage_run.py; fi
$P $R scripts/chain_clock.py $TD $O --ns 6 --nt $NT --objective huber --jit 25 --ncopy 2 --npz ${O}_HuJ.npz $XA | tail -1
leave_stage_noa1=1 $P $R scripts/chain_leave.py ${O}_HuJ.npz --hnpz ${O}_HuJ.npz --tag $T --ns 3 --nt $NT | grep -E "^\s+A0\+L " | cut -c1-120
$P scripts/stage12_a0l.py ${O}_HuJ.npz $T ${O}_HA0L >/dev/null; ls ${O}_HA0L_*.npy | wc -l
