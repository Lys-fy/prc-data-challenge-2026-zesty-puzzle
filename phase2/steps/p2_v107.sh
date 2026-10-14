#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# FORM=v107: the chain-variant levers v100 -> v102 -> F2K -> v106b -> v106K -> TW1' -> KRS / KRS-NL, BY RULE, on top of a FORM=v99 phase2
# workspace (phase2/run_pipeline.sh). Each lever is an isolate delta on the previous step's workspace output; no earlier submission is read.
#   usage: bash phase2/steps/p2_v107.sh <period> [step ...]        (markers logs/<step>.done, logs logs/<step>.log, step times logs/times.tsv)
# Composition:
#   v100  = iso(final.v99 ; I(nbr_clock_twin_bm) - I(NBR_CLOCK_TWIN))          NBR_CLOCK MBm member as xtd2 (Kq, NNA3), same-recipe MA twin
#   v102  = iso(v100      ; I(kcatlg_stack)   - I(XB))            KCATLg (log-target Kumo member with tail guard) as xtd2; twin = the WIDELEAF chain (phase2 iso_XB).
#                                                       V102ARM=k250_stack puts K250 (KSBWPc100) in this slot instead.
#   F2K   = iso(v102      ; I(KSTK_CLOCK)  - I(k250_stack))           K250 visible to the Hu6/A0L clock stages (kumo_clock_chain.sh, reuses the k250_stack stack/C4/treeD)
#   v106b = iso(F2K       ; I(BIG_NOVELTY) - I(k250_stack))           BIG set transformer (big_big3 / big_bigp x3) as xtd3 with the novelty mask
#   v106K = iso(v106b     ; I(kstk_input)   - I(k250_stack))           KSTK: K250 as chain-stacker augment (gK = data/preds/KSTK6<C>P.npy)
#   TW1'  = +TW1S s (1264, a 2025 constant) on the LFPG twin rows (ids: TW1IDS; MODE=new: rebuilt from data/raw by rule, step tw1ids)
#   v107  = KRS(v106K; beta by K250-spread tercile, coef v106K_K2_KRS_prod_coef.json (2025 fit)) with the TW1' rows kept at base + TW1S
#   I(X) = the v88-line post of the chain set (LdTK, X Kq, X NNA3) = phase2 post_iso: w5 pool -> (B) -> U3 stitch -> standfloor -> offblock_pull -> px_safe (NEWSTAND_DELTA off)
#   iso = scripts/isolate_delta.py (unmatched + exposed rows protected, LOBT law +-3606, floor 60, rint, validator) with BASEMD5 empty.
# Fold sides (2025 only) are linked from the project (step v107link, FOLD_LINK); every production stage refits its small model on them at run time.
set -euo pipefail
P2=$(cd "$(dirname "$0")/.." && pwd); REPO=${PRC_ROOT:-$(dirname "$P2")}; PERIOD=${1:?usage: p2_v107.sh <period> [step ...]}; shift
set -a; source "$P2/periods/$PERIOD.env"; set +a
WS=$P2/work/${WS_NAME:-$PERIOD}; PY=$REPO/.venv/bin/python; NT=${NT:-6}; NTC=${NT_V107:-4}
export P2_REPO=$REPO P2_ROOT=$WS PRC_ROOT=$WS P2_LOG=$WS/logs/ledger.jsonl P2_STRICT=${P2_STRICT:-1} PYTHONPATH=$P2/lib/guard daystate_root=$WS daystate_dir=$WS/scratchpad/xl
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage P2_NRANK=${NRANK:-344841} PYTHONHASHSEED=0
[ "$MODE" = new ] && export P2_MODE=new
S=submissions; Q=scratchpad/xpr; AX=scratchpad/xau1; H=scratchpad/xhook
TW1S=${TW1S:-1264}; KRSCOEF=${KRSCOEF:-scratchpad/xchain2/v106K_K2_KRS_prod_coef.json}
TW1IDS=${TW1IDS:-scratchpad/xfinal/tw1p_ids.parquet}   # step tw1ids (rule); MODE=repro also asserts it equals the Jan/Jul list scratchpad/xquick/tw1p_ids_2026.parquet
NOVELTY=${NOVELTY:-$WS/scratchpad/xds2/$([ "$MODE" = new ] && echo novelty_mask_final.npy || echo novelty_mask_ids.npy)}   # MODE=new: step novelty
log(){ echo "[$(date +%T)] $*"; }
run(){ local s=$1; shift; [ -f "$WS/logs/$s.done" ] && { log "skip $s (done)"; return 0; }
       log "step $s"; local t0; t0=$(date +%s)
       ( cd "$WS" && "$@" ) > "$WS/logs/$s.log" 2>&1 || { log "FAILED $s (see $WS/logs/$s.log)"; tail -20 "$WS/logs/$s.log"; exit 1; }
       grep -q "^Traceback (most recent call last)" "$WS/logs/$s.log" && { log "FAILED $s (Python traceback in $WS/logs/$s.log)"; grep -A3 "^Traceback" "$WS/logs/$s.log" | tail -20; exit 1; }
       printf "%s\t%s\t%s\t%s\n" "$s" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u +%FT%TZ)" "$(( $(date +%s) - t0 ))" >> "$WS/logs/times.tsv"
       touch "$WS/logs/$s.done"; log "done $s ($(( $(date +%s) - t0 )) s)"; }

step_v107link(){   # 2025 fold sides of every chain of the line (project files, read-only links; copy-on-write in the workspace) + 2025 constants
  env -u PYTHONPATH -u P2_ROOT -u P2_STRICT $PY - "$REPO" "$WS" "$MODE" <<'PY'
import os, re, sys
R, W, MODE = sys.argv[1:4]
fold = re.compile(r"(_janjul|_months\d+)(\.npy|\.parquet|\.npz|_)|_HuJ\.npz$")
bad = re.compile(r"^PR_|prod|rank|cand|2026|^prep_")
def link(src, dst):
    if os.path.lexists(dst): return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True); os.symlink(os.path.realpath(src), dst); return 1
n = {}
# v87 chains (p13stk + fold stage), WIDELEAF/PSEUDOLABEL chains, every v100+ chain prefix
for d, pat in (("p13", r"^P13_KB"), ("xau1", r"^clockfold1_"), ("xlv", r"^leave_stage_xau1_"), ("xhook", r"^P13_"), ("xbase", r"^P13_XB"), ("xsyn", r"^P13_XS")):
    k = 0
    for f in sorted(os.listdir(f"{R}/scratchpad/{d}")):
        if re.search(pat, f) and fold.search(f) and not bad.search(f): k += link(f"{R}/scratchpad/{d}/{f}", f"{W}/scratchpad/{d}/{f}")
    n[d] = k
# 2025 constants at the workspace paths the scripts read: KRS beta and KRS-NL coefficients (phase2/params), the KRS-NL LightGBM model
# (make constants: scratchpad/xkrsnl/prod/gbm_A.txt, not published); in MODE=repro also the period id lists of the Jan/Jul build
const = {"scratchpad/xchain2/v106K_K2_KRS_prod_coef.json": "phase2/params/krs_coef.json", "scratchpad/xkrsnl/prod/coef.json": "phase2/params/krsnl_coef.json",
         "scratchpad/xkrsnl/prod/gbm_A.txt": "scratchpad/xkrsnl/prod/gbm_A.txt"}
if MODE != "new": const.update({c: c for c in ("scratchpad/xquick/tw1p_ids_2026.parquet", "scratchpad/xds2/novelty_mask_ids.npy", "scratchpad/xhr/exposed_rows.parquet")})
n["const"] = sum(link(f"{R}/{src}", f"{W}/{dst}") for dst, src in const.items())
print("v107link:", n)
PY
  for C in dTK Kq NNA3; do   # FOLD_LINK for the v87 chains: their stack (p13stk) and fold stage (fold_<C>.v87) are 2025 fits -> linked above
    ls scratchpad/p13/P13_KB${C}xSTK_janjul.npy scratchpad/p13/P13_KB${C}xTD_janjul.npy $AX/clockfold1_l${C}_HuJ.npz $AX/clockfold1_l${C}_HA0L_janjul.npy > /dev/null
    if [ "${FOLD_LINK:-1}" = 1 ]; then touch logs/p13stk_$C.done logs/fold_$C.v87.done; fi; done
  mkdir -p scratchpad/xfinal scratchpad/xds2 scratchpad/xquick scratchpad/xhr scratchpad/xchain2 scratchpad/xchain
}
chain(){   # $1 prefix, $2 chain (Kq|NNA3): production side (pre -> C4 -> treeD -> Hu6 -> A0+L) on the period's ranking rows
  local X=$1 C=$2; export NT=$NTC
  case $X in
    nbr_clock_twin_bm) KM=nbr_clock_mbm KMP=nbr_clock_mbmp PFX=nbr_clock_twin_bm bash scripts/chainvar_prod.sh $C;;
    NBR_CLOCK_TWIN)  KM=nbr_clock_ma  KMP=nbr_clock_map  PFX=NBR_CLOCK_TWIN  bash scripts/chainvar_prod.sh $C;;
    k250_stack)   KM=KSBWPc100 KMP=KSBWPc100P PFX=k250_stack   bash scripts/chainvar_prod.sh $C;;
    KSTK_CLOCK)  KM=KSBWPc100 KMP=KSBWPc100P PFX=k250_stack   bash scripts/kumo_clock_chain.sh $C;;
    kcatlg_stack)   KM=KCATLg    KMP=KCATLgP    PFX=kcatlg_stack   bash scripts/chainvar_prod.sh $C;;   # KCATLg (scripts/kumo_kcat_logtarget_*)
    BIG_NOVELTY) PFX=BIG_NOVELTY big=big_big3 BIG_PROD_PREFIX=big_bigp extra3_mask= extra3_mask_ids=$NOVELTY bash scripts/dayset_chain.sh prod $C;;
    kstk_input)   bash scripts/kstk_run.sh prod $C;;
    *) echo "unknown chain $X"; return 2;;
  esac
  local F=$X; [ "$X" = KSTK_CLOCK ] && F=k250_stackC; ls -la $H/PR_${F}${C}_XKAU1_HA0L.parquet
}
iso(){     # $1 prefix: the v88-line post of (LdTK, $1 Kq, $1 NNA3) = run_pipeline.sh post_iso (same rules, same fixed arms) -> submissions/v107_iso_$1.parquet
  local T=$1; local F=$1; [ "$T" = KSTK_CLOCK ] && F=k250_stackC   # kumo_clock_chain.sh writes PR_<PFX>C<chain>_XKAU1 (PFX k250_stack)
  local C2=$H/PR_${F}Kq_XKAU1_HA0L.parquet C3=$H/PR_${F}NNA3_XKAU1_HA0L.parquet POOL=$Q/pool.v107$T.parquet B=$Q/pool.v107$T.B.parquet P=$Q/post.v107$T
  env W=0.0,0.3212,0.3065,0.0,0.3722 $PY scripts/pool_mix.py stages/unmatched_rule.parquet $POOL $AX/PR_LdTK_AMXAU1_HA0L.parquet $AX/PR_LdTK_AMXAU1_HA0L.parquet $C2 $AX/PR_LdTK_AMXAU1_HA0L.parquet $C3
  if [ "${PRU:-1}" = 1 ]; then
    $PY scripts/pru_daily_calib.py --pool $AX/clockfold1_ldtk_ha0l,$AX/clockfold1_${T}Kq_HA0L,$AX/clockfold1_${T}NNA3_HA0L --w 0.3212,0.3065,0.3723 --out $Q/pru.v107$T --um25 scratchpad/xau3/um25_A1.npz \
      --prod $POOL --prod-out $B --prod-arm B --prod-um prod --um-src stages/unmatched_link_a1.parquet
  else cp $POOL $B; fi
  $PY "$P2/steps/p2_slot.py" --out $P.stitch.parquet --unmatched U3 --matched $B
  cp "$P2/${POSTRULES_PARAMS:-params/postrules_fixed.json}" scratchpad/p2/postrules.v107.json
  $PY "$P2/steps/p2_postrules.py" apply --in $P.stitch.parquet --out $P.xsx.parquet --params scratchpad/p2/postrules.v107.json --only xsx
  $PY "$P2/steps/p2_postrules.py" apply --in $P.xsx.parquet --out $P.xpk.parquet --params scratchpad/p2/postrules.v107.json --only xpk
  $PY "$P2/steps/p2_exposed_scenarios.py" --in $P.xpk.parquet --out $S/v107_iso_$T.parquet --ledger scratchpad/p2/xhr_ledger.v107$T.parquet
  $PY "$P2/steps/p2_check.py" validate $S/v107_iso_$T.parquet
}
isolate(){ # $1 out, $2 arm iso, $3 twin iso, $4 base
  rm -f $1; env BASEMD5= VN=107 $PY scripts/isolate_delta.py $1 $2 $3 $4 2>&1 | grep -vE "polars|implode|is_in"; ls -la $1; }
step_exposed(){    # MODE=new: the exposed class by rule (unmatched ranking rows with MVT - SCHED > 8 h), the protected set of every isolate
  env P2_STRICT=0 $PY - <<'PY'
import polars as pl
F = pl.read_parquet("data/features_v9.parquet", columns=["MVT_ID_mvt", "is_rank", "unmatched", "mvt_sched"]).filter(pl.col("is_rank") == 1)
E = F.filter((pl.col("unmatched") == 1) & (pl.col("mvt_sched") > 28800)).select("MVT_ID_mvt")
E.write_parquet("scratchpad/xhr/exposed_rows.parquet"); print("exposed rows (rule):", E.height)
PY
}
step_novelty(){    # MODE=new: BIG_NOVELTY novelty mask for the period (label-free; the shipped 2025 part is kept so the linked fold side is unchanged)
  env P2_STRICT=0 $PY "$P2/steps/p2_novelty.py" "$REPO/scratchpad/xds2/novelty_mask_ids.npy" scratchpad/xds2/novelty_mask_final.npy
}
step_tw1ids(){     # TW1' class by rule from the raw ranking file: LFPG, same non-null callsign, a pair of exactly 2 DEP records,
                   # |dMVT| <= 300 s, exactly one NM-matched record -> that record. Label-free; no ordering is used.
  env P2_STRICT=0 $PY "$P2/steps/p2_tw1_rule.py" data/raw/ranking.parquet data/features_v9.parquet $TW1IDS $([ "$MODE" = new ] || echo "--expect scratchpad/xquick/tw1p_ids_2026.parquet")
}
step_v100(){  isolate $Q/v107_v100.parquet  $S/v107_iso_nbr_clock_twin_bm.parquet $S/v107_iso_NBR_CLOCK_TWIN.parquet $S/final.v99.parquet; }
step_v102(){  isolate $Q/v107_v102.parquet  $S/v107_iso_${V102ARM:-kcatlg_stack}.parquet   $S/v99_iso_XB.parquet     $Q/v107_v100.parquet; }
step_f2k(){   isolate $Q/v107_f2k.parquet   $S/v107_iso_KSTK_CLOCK.parquet  $S/v107_iso_k250_stack.parquet   $Q/v107_v102.parquet; }
step_v106b(){ isolate $Q/v107_v106b.parquet $S/v107_iso_BIG_NOVELTY.parquet $S/v107_iso_k250_stack.parquet   $Q/v107_f2k.parquet; }
step_v106k(){ isolate $Q/v107_v106k.parquet $S/v107_iso_kstk_input.parquet   $S/v107_iso_k250_stack.parquet   $Q/v107_v106b.parquet; }
step_krs(){   # TW1' rows = base + TW1S (kept out of KRS), all other matched unprotected rows: KRS toward K250
  env TW1S=$TW1S TW1IDS=$TW1IDS $PY - <<'PY'
import os, polars as pl
c = "TAXITIME_SEC_mvt"; B = pl.read_parquet("scratchpad/xpr/v107_v106k.parquet"); ids = pl.read_parquet(os.environ["TW1IDS"])["MVT_ID_mvt"].cast(pl.Int64)
s = int(os.environ["TW1S"]); A = B.with_columns(pl.when(pl.col("MVT_ID_mvt").cast(pl.Int64).is_in(ids.implode())).then(pl.col(c) + s).otherwise(pl.col(c)).cast(B[c].dtype))
n = int((A[c] != B[c]).sum()); print(f"TW1' +{s} s on {n} rows (ids {ids.len()})"); assert n == ids.len(); A.write_parquet("scratchpad/xpr/v107_v106k_tw1p.parquet")
PY
  rm -f $S/final.v107.parquet
  env TW1ADD=$Q/v107_v106k_tw1p.parquet:$Q/v107_v106k.parquet BASEMD5= VN=107 $PY scripts/krs_build.py $Q/v107_v106k.parquet $TW1IDS $KRSCOEF $S/final.v107.parquet 2>&1 | grep -vE "polars|implode|is_in"
  $PY "$P2/steps/p2_check.py" validate $S/final.v107.parquet; md5sum $S/final.v107.parquet
}
step_krsnl(){  # v111 = the same base re-stepped by the non-linear KRS arm A (KRS_NONLINEAR: v107 tercile line + 0.5 g(line, K-line, spread, apt, hour), 2025 fit)
  rm -f $S/final.v111.parquet
  env TW1ADD=$Q/v107_v106k_tw1p.parquet:$Q/v107_v106k.parquet BASEMD5= VN=111 $PY scripts/krs_nonlinear.py build ${KRSMODE:-A} $Q/v107_v106k.parquet $TW1IDS $S/final.v111.parquet 2>&1 | grep -vE "polars|implode|is_in"
  $PY "$P2/steps/p2_check.py" validate $S/final.v111.parquet; md5sum $S/final.v111.parquet
}
EXP=$([ "$MODE" = new ] && echo "exposed novelty" || true)
DEF="v107link c_nbr_clock_twin_bm_Kq c_nbr_clock_twin_bm_NNA3 c_NBR_CLOCK_TWIN_Kq c_NBR_CLOCK_TWIN_NNA3 c_k250_stack_Kq c_k250_stack_NNA3 c_KSTK_CLOCK_Kq c_KSTK_CLOCK_NNA3 c_BIG_NOVELTY_Kq c_BIG_NOVELTY_NNA3 c_kstk_input_Kq c_kstk_input_NNA3 c_kcatlg_stack_Kq c_kcatlg_stack_NNA3 i_nbr_clock_twin_bm i_NBR_CLOCK_TWIN i_k250_stack i_KSTK_CLOCK i_BIG_NOVELTY i_kstk_input i_kcatlg_stack $EXP v100 v102 f2k v106b v106k tw1ids krs krsnl"
for s in ${*:-$DEF}; do
  case $s in
    v107link) run $s step_v107link;;
    c_*) X=${s#c_}; run $s chain ${X%_*} ${X##*_};;
    i_*) run $s iso ${s#i_};;
    exposed|novelty|tw1ids|v100|v102|f2k|v106b|v106k|krs|krsnl) run $s step_$s;;
    *) echo "unknown step $s"; exit 2;;
  esac
done
