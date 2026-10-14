#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# v107 fold line and its KRS coefficients: (B) rung of each chain -> additive line at (B) -> post rules -> KRS fit (krs_fit2.py MODES=KRS).
#   line = chain(k250_stack) + (chain(KSTK_CLOCK) - chain(k250_stack)) + (chain(BIG_NOVELTY) - chain(k250_stack)) + (chain(kstk_input) - chain(k250_stack))     (6 folds, [id, pred])
#   The shipped line was assembled in steps: K250 in the clock stages (KSTK_CLOCK) and the first day-set chain on chain(k250_stack), KRS refit;
#   then the day-set chain replaced by BIG_NOVELTY (delta against the first day-set chain), KRS refit; then KSTK (kstk_input against chain(k250_stack)), KRS refit.
#   The first day-set chain enters once with + and once with -, so the line is written here with every arm against chain(k250_stack).
# Each (B) vector = w5 pool (LdTK, <C>Kq, <C>NNA3) -> pru_daily_calib (B) um25_A1, skipped when it exists (override a prefix with env KSTK_B / KSTK_CLOCK_B / BIG_NOVELTY_B / KSTK_STACKER_B).
# Inputs: scratchpad/xau1/clockfold1_{LdTK,k250_stackkq,KSTK_NNA3_FOLD,kstk_clockkq,KSTK_CLOCK_NNA3_FOLD,big_noveltykq,BIG_NOVELTY_NNA3_FOLD,kstk_inputkq,KSTK_STACKER_NNA3}_HA0L_<fold>.npy (A2 chain fold sides), scratchpad/xau3/um25_A1.npz,
#         data/oof/KSBWPc100_<fold>.npy (K250 and its spread).
# usage: [NAME=v106K] [O=scratchpad/xchain2] NT=4 bash scripts/krs_fold_line.sh
# Outputs $O/<NAME>_{LE3B,POST,K2_KRS}_<fold>.npy and $O/<NAME>_K2_KRS_prod_coef.json (the KRS beta read by krs_build.py / krs_nonlinear.py).
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; A=scratchpad/xau1; O=${O:-scratchpad/xchain2}; NAME=${NAME:-v106K}; mkdir -p $O
NT=${NT:-4}; export OMP_NUM_THREADS=$NT POLARS_MAX_THREADS=$NT OPENBLAS_NUM_THREADS=$NT
W5=0.3212,0.3065,0.3723; UM=scratchpad/xau3/um25_A1.npz
KSTK_B=${KSTK_B:-$O/chain_fold_line_xks_w5_le3b}; KSTK_CLOCK_B=${KSTK_CLOCK_B:-$O/chain_fold_line_xks_c_le3b}; BIG_NOVELTY_B=${BIG_NOVELTY_B:-$O/big_novelty_le3b}; KSTK_STACKER_B=${KSTK_STACKER_B:-$O/chain_fold_line_xkb_w5_le3b}
echo "krs_fold_line $NAME -> $O host $(hostname) $(date -u +%FT%TZ)"
calb() { [ -f ${1}_janjul.npy ] || $P scripts/pru_daily_calib.py --pool $A/clockfold1_ldtk_ha0l,$A/clockfold1_${2}Kq_HA0L,$A/clockfold1_${2}NNA3_HA0L --w $W5 --out ${1%B} --um25 $UM | grep -E "INFO B " | head -2; }
calb $KSTK_B k250_stack; calb $KSTK_CLOCK_B KSTK_CLOCK; calb $BIG_NOVELTY_B BIG_NOVELTY; calb $KSTK_STACKER_B kstk_input
$P - "$O/${NAME}_LE3B" "$KSTK_B" "$KSTK_CLOCK_B" "$BIG_NOVELTY_B" "$KSTK_STACKER_B" <<'PY'
import sys, numpy as np
out, ref, *arms = sys.argv[1:]
for t in ["months39", "months28", "months410", "janjul", "months612", "months511"]:
    R = np.load(f"{ref}_{t}.npy"); o = np.argsort(R[0]); ids = R[0][o]; r = R[1][o]; v = r.copy()
    for a in arms:
        Z = np.load(f"{a}_{t}.npy"); oz = np.argsort(Z[0]); assert np.array_equal(Z[0][oz], ids), (a, t)
        v += Z[1][oz] - r
    np.save(f"{out}_{t}.npy", np.vstack([ids, v])); print(f"{t}: line - chain(k250_stack) rms {np.sqrt(np.mean((v - r) ** 2)):.2f} s on {len(ids):,} rows")
PY
$P scripts/chain_post.py $O/${NAME}_LE3B $O/${NAME}_POST | tail -1
MODES=KRS $P scripts/krs_fit2.py $O/${NAME}_POST $O/${NAME}_K2 | grep -E "^\[(prod|beta)"
echo KRS_FOLD_LINE_DONE
