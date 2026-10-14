#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KSTK chain variant: Kumo Tabular as a second stacking engine on the chain stacker's meta frame. Scripts: kstk_stack.py, kstk_assemble.py,
# kstk_bundle_prod.sh; the chain-variant scripts chainvar_stk.sh / chainvar_fold.sh are called unchanged (they read env STACKSRC).
# Member = KSBWPc100 (K250), chain prefix of the KB chains = kstk_input, Kumo run name KSTK6<C> (GPU step between dump and gk).
# usage (CPU host, $PRC_ROOT): NT=8 bash scripts/kstk_run.sh <step> [Kq NNA3]   steps (in order):
#   dump     chain-stacker meta frames: 6 folds (chainvar_stk.sh STKONLY, STACKSRC=kstk_stack.py KSTK_DUMP) + prod (kstk_bundle_prod.sh KSTK_DUMP)
#            -> kumo_gpu/data/payload/stk6/stk_<C>_{6 folds,prod}.npz + MD5SUMS; adapter selftest --real
#   regress  default-path regression: chainvar_stack.py stack (PFX kstk_regress) must be byte-identical to the dump run's stack (PFX kstk_dump, kstk_stack + KSTK_DUMP)
#   gk RUN   after the Kumo GPU run: instance files of <RUN> -> data/oof/KSTK6<C>_<fold>.npy, data/preds/KSTK6<C>P.npy (scripts/kstk_gk.py, validated)
#   fold     KB chain fold side: chainvar_fold.sh with STACKSRC=kstk_stack.py KSTK_GK=KSTK6<C> (stack -> C4/tree D -> Huber clock stage -> A0+L) -> clockfold1_xkb<C>_*
#   prod     KB chain prod side: kstk_bundle_prod.sh with KSTK_GKP=data/preds/KSTK6<C>P.npy (chainvar_prod.sh settings) -> PR_XKB<C>_XKAU1_HA0L
#   gate     w5(LdTK, kstk_inputkq, KSTK_STACKER_NNA3) -> pru_daily_calib (B) um25_A1 vs chain(k250_stack) (REF, default scratchpad/xcpu2/chain_fold_line_xks_w5_le3b) : gate_unsealed_folds + WD + fresh 511/612
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; P=.venv/bin/python; ST=$1; shift; NT=${NT:-8}; export NT
export KM=KSBWPc100 KMP=KSBWPc100P; KD=kumo_gpu/data/payload/stk6; mkdir -p $KD scratchpad/xkstk; TV="/usr/bin/time -f kstk_assemble_time_%C_wall=%es_maxrss=%MkB"
PENV="pru=1 FORM=PC A1=1 FBMODE=member ARM=KUMO_CLOCKFOLD_ARM ARMIN=XK SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1"   # = chainvar_prod.sh
echo "KSTK_ASSEMBLE $ST host $(hostname) $(date -u +%FT%TZ) NT $NT"; md5sum scripts/kstk_stack.py scripts/kstk_assemble.py scripts/kstk_bundle_prod.sh scripts/chainvar_stack.py scripts/chainvar_assemble.py | cut -c1-12,34-
case $ST in
dump) for C in ${@:-Kq NNA3}; do
    [ -f $KD/stk_${C}_janjul.npz ] || KSTK_DUMP=$KD KSTK_CHAIN=$C STACKSRC=scripts/kstk_stack.py STKONLY=1 pru=1 KM=$KM PFX=kstk_dump $TV bash scripts/chainvar_stk.sh $C
    [ -f $KD/stk_${C}_prod.npz ] || { PFX=kstk_dump $P scripts/chainvar_check.py inputs prod $C | grep -v "kstk_dump" | tail -4 || true; env KSTK_DUMP=$KD/stk_${C}_prod.npz KM=$KM KMP=$KMP PFX=kstk_dump $PENV $TV bash scripts/kstk_bundle_prod.sh $C; }
  done
  (cd $KD && md5sum stk_*.npz > MD5SUMS && cat MD5SUMS); du -sh $KD
  KSTK_DIR=$KD $P kumo_gpu/cluster/km_adapters_kstk.py --selftest --real
  $P - <<'EOF'
import numpy as np, os   # info: are the chain-stacker frames on the 4 gate folds the KSTK_FEED (CAPACITY_COMBINE harness) frames?
for c in ("Kq", "NNA3"):
    for t in ("months39", "months28", "months410", "janjul"):
        a, b = f"kumo_gpu/data/payload/stk6/stk_{c}_{t}.npz", f"kumo_gpu/data/payload/stk/stk_{c}_{t}.npz"
        if not (os.path.exists(a) and os.path.exists(b)): continue
        A, B = np.load(a), np.load(b); ca, cb = list(A["cols"]), list(B["cols"]); com = np.intersect1d(A["mid"], B["mid"])
        ia = np.searchsorted(np.sort(A["mid"]), com); oa = np.argsort(A["mid"])[ia]; ib = np.argsort(B["mid"])[np.searchsorted(np.sort(B["mid"]), com)]
        XA = A["X"][oa][:, [ca.index(x) for x in cb if x in ca]]; XB = B["X"][ib][:, [i for i, x in enumerate(cb) if x in ca]]
        d = np.abs(np.nan_to_num(XA, nan=-9e9) - np.nan_to_num(XB, nan=-9e9)); dv = np.abs(A["v41"][oa] - B["v41"][ib])
        print(f"FRAMECHK {c} {t}: rows chain {len(A['mid']):,} xfeed {len(B['mid']):,} common {len(com):,} | cols same {ca == cb} | max|dX| {d.max():.4g} (cols differing {[cb[j] for j in np.where(d.max(0) > 1e-3)[0]][:6]}) | max|dv41| {dv.max():.4g}")
EOF
  ;;
regress) for C in ${@:-Kq NNA3}; do
    STKONLY=1 pru=1 KM=$KM PFX=kstk_regress $TV bash scripts/chainvar_stk.sh $C
    for t in months39 months28 months410 months612 months511 janjul; do a=$(md5sum < scratchpad/xhook/P13_XKR${C}xSTK_$t.npy); b=$(md5sum < scratchpad/xhook/P13_XKD${C}xSTK_$t.npy)
      [ "$a" = "$b" ] && echo "REGRESS $C $t IDENTICAL ${a:0:8}" || echo "REGRESS $C $t DIFFERENT ${a:0:8} ${b:0:8}"; done
    # numeric view + the run-to-run baseline: kstk_stack with NO knob (PFX kstk_noknob) vs the chainvar_stack.py stack (kstk_regress) and vs the dump run (kstk_dump)
    STKONLY=1 pru=1 KM=$KM PFX=kstk_noknob STACKSRC=scripts/kstk_stack.py $TV bash scripts/chainvar_stk.sh $C
    $P - $C <<'EOF2'
import sys, numpy as np, polars as pl
C = sys.argv[1]; F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "y"]); fy = dict(zip(F["MVT_ID_mvt"].cast(pl.Float64).to_list(), F["y"].to_list()))
for t in ("months39", "months28", "months410", "months612", "months511", "janjul"):
    L = {k: np.load(f"scratchpad/xhook/P13_{k}{C}xSTK_{t}.npy") for k in ("kstk_regress", "kstk_noknob", "kstk_dump")}
    y = np.array([fy[i] for i in L["kstk_regress"][0]], float); assert all(np.array_equal(L[k][0], L["kstk_regress"][0]) for k in L)
    msg = " | ".join(f"{a}~{b} max|d| {np.abs(L[a][1]-L[b][1]).max():.4f} dMSE {np.mean((L[a][1]-y)**2)-np.mean((L[b][1]-y)**2):+.3f}" for a, b in (("kstk_noknob", "kstk_regress"), ("kstk_dump", "kstk_regress"), ("kstk_dump", "kstk_noknob")))
    print(f"REGNUM {C} {t}: {msg}", flush=True)
EOF2
    rm -f scratchpad/xhook/P13_XKR${C}xSTK_*.npy scratchpad/xhook/P13_XKN${C}xSTK_*.npy
  done;;
gk) RUN=$1; shift; $P scripts/kstk_gk.py $RUN ${@:-Kq NNA3} | tee scratchpad/xkstk/gk.txt;;
fold) for C in ${@:-Kq NNA3}; do
    for t in months39 months28 months410 months612 months511 janjul; do ls data/oof/KSTK6${C}_$t.npy >/dev/null; done
    [ -f scratchpad/xau1/clockfold1_xkb${C}_HA0L_janjul.npy ] && { echo "fold $C done (kept)"; continue; }
    STACKSRC=scripts/kstk_stack.py KSTK_GK=KSTK6$C KM=$KM PFX=kstk_input $TV bash scripts/chainvar_fold.sh $C
  done;;
prod) for C in ${@:-Kq NNA3}; do
    [ -f scratchpad/xhook/PR_XKB${C}_XKAU1_HA0L.parquet ] && { echo "prod $C done (kept)"; continue; }
    ls data/preds/KSTK6${C}P.npy >/dev/null; PFX=kstk_input $P scripts/chainvar_check.py inputs prod $C
    env KSTK_GKP=data/preds/KSTK6${C}P.npy KM=$KM KMP=$KMP PFX=kstk_input $PENV $TV bash scripts/kstk_bundle_prod.sh $C
  done;;
gate) A=scratchpad/xau1; O=scratchpad/xkstk; REF=${REF:-scratchpad/xcpu2/chain_fold_line_xks_w5_le3b}; W5=0.3212,0.3065,0.3723; UM=scratchpad/xau3/um25_A1.npz
  ls ${REF}_janjul.npy ${REF}_months511.npy >/dev/null
  [ -f $O/chain_fold_line_xkb_w5_le3b_janjul.npy ] || $P scripts/pru_daily_calib.py --pool $A/clockfold1_ldtk_ha0l,$A/clockfold1_xkbkq_ha0l,$A/clockfold1_xkbnna3_ha0l --w $W5 --out $O/chain_fold_line_xkb_w5_le3 --um25 $UM | grep -E "INFO B " | head -2
  $P scripts/gate_unsealed_folds.py "GATE_KSTK=npy:$O/chain_fold_line_xkb_w5_le3b~npy:$REF" | grep -E "^\[" | tee $O/gate_KSTK.txt
  $P scripts/gate_within_day.py $O/chain_fold_line_xkb_w5_le3b $REF $O/chain_fold_line_kstk months39,months28,months410,janjul | tail -4 | tee -a $O/gate_KSTK.txt
  $P scripts/gate_fresh_folds.py "FRESH_KSTK=npy:$O/chain_fold_line_xkb_w5_le3b~npy:$REF" | grep -E "^\[|MW" | tee -a $O/gate_KSTK.txt
  for C in Kq NNA3; do [ -f $A/clockfold1_xks${C}_HA0L_janjul.npy ] && $P scripts/gate_unsealed_folds.py "info_${C}_HA0L=npy:$A/clockfold1_xkb${C}_HA0L~npy:$A/clockfold1_xks${C}_HA0L" | grep -E "^\[" | tee -a $O/gate_KSTK.txt; done
  ;;
*) echo "step?"; exit 1;;
esac
echo "kstk_assemble_done $ST $(date -u +%FT%TZ)"
