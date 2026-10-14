#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# phase2_ws: phase-2 pipeline, BY RULE, for one period.   usage: phase2/run_pipeline.sh <period> [step ...]
#   steps (default: all, in this order; each step is skipped when logs/<step>.done exists, delete the marker to rerun):
#     ws          create / refresh the workspace phase2/work/<period> (frozen script copy + link farms, p2ws.py)
#     np_features MODE=new, real new period only: official ranking file -> features (build_features.py --rank)
#     np_side     MODE=new: per-batch side tables (dreg, xba fallback ids), period-input coverage check, q (fallback_classifier rank), XQ nested q table
#     m_<family>  MODE=new: retrain one member family of manifest/members.tsv on the period (seeds MSEEDS, default the v76 seeds)
#     np_derive   MODE=new: derived members (queue_decomp_ebdt; v87: sixseed_chain_ebdt6, sixseed_chain_nnra3x6) + seed aliasing for seeds not retrained
#     np_xau2     MODE=new, v87 form: NEWSTAND_DELTA new-stand mapping + delta members
#     pre         U (unmatched rules) + stage files stages/unmatched_rule, stages/fallback_base (fallback rows) + unm_resid model -> stages/unmatched_resid
#     fold_dTK fold_Kq fold_NNA3   fold side of each pool chain (stage fits on the 2025 month-fold OOF)
#     prod_dTK prod_Kq prod_NNA3   production side on the period's ranking rows   (chain_<C> = fold_<C> + prod_<C>)
#     post        pool (pool_mix) -> (B) PRU daily calibration -> unmatched stitch -> standfloor floor -> offblock_pull overlay -> px_safe PX-safe -> validate
#     l0check     layer-0 check: features rebuilt from data/raw by build_features.py vs data/features_v9.parquet
#     compare     comparison of the final workspace file with a reference file (REF) if set
#   FORM (period file; default px) selects the chain form of the build line:
#     px  = pooled chains with the PRU daily inputs (xpr) + (B) + standfloor / offblock_pull fitted + px_safe
#     t2x = chains with stack inputs K(b) + xpr + A1, fallback rows from the member; post rules at the fixed arms of params/postrules_fixed.json
#     v87 = chain fold TD + clock stage with adsb5 / adsb6 / adsbL clocks, six-seed production (6-seed members, 6 meta seeds, 6 L seeds);
#           unmatched: unm_resid -> callsign link A1 -> V3; (B) day sums with the A1 values; new-stand delta
#     v99 = the base line of the final period: the v87 chains (same steps and markers as FORM=v87) without the new-stand delta,
#           then sequentially, each on the previous step's workspace output (no uploaded file is read):
#             iso_L   post line of the v87 chains (pool -> (B) -> unmatched U3 = unm_resid -> callsign link A1 -> V3 -> standfloor -> offblock_pull -> px_safe)
#             WIDELEAF=1 iso_XB: the same post line with the wide-leaf Kq / NNA3 chains (1,023-leaf member as extra stacker input)
#             PSEUDOLABEL=1  iso_XS: the same with the pseudo-label Kq / NNA3 chains (pseudo-labelled L member, teacher = iso_L of the same period).
#                     MODE=new: two passes (iso_L -> xs_pseudo -> m_pseudolabel_pl3p88 -> xs_prod_* -> iso_XS); default 0, 1 in phase2/periods/final.env
#             compose: iso_L + (iso_XB - iso_L) + (iso_XS - iso_L)        E1S=1  E1s rule on exposed non-LIRF rows (default 0)
#             F13=1   Kumo F1 / F3 on unm_other (kumo_icl_unmatched.py F1, F3 -> kumo_icl_build.py UL,UU on the workspace file)
#             F4=1    Kumo F4 on unm_LIRF (kumo_icl_lirf.py F4 -> BUILD on the F1F3 file)
#           Steps: v99prep kumo_f13 kumo_f4 [xb_fold_<C> xb_prod_<C>] iso_L [xs_pseudo m_pseudolabel_pl3p88] [xs_fold_<C> xs_prod_<C>] iso_XB iso_XS post compare
#           (C in Kq, NNA3; xb_fold / xs_fold link the 2025-fitted fold side when FOLD_LINK=1, the default, and rebuild it when FOLD_LINK=0).
#   Flags (period file or environment): NEWSTAND_DELTA (default 1 for v87, 0 otherwise), E1S, WIDELEAF, PSEUDOLABEL (default 0), F13, F4 (default 1), FOLD_LINK (1).
#   Form-specific steps (fold_*, prod_*, pre_v87, post, compare) write markers <step>.<form>; shared steps (ws, pre, p13stk_*, np_*, m_*) plain.
#   FORM=v99 shares the chain markers of v87 (fold_<C>.v87, prod_<C>.v87); its own steps (v99prep, kumo_*, xb_*, xs_*, iso_*) are plain.
# Every Python process runs under the workspace guard (lib/guard/sitecustomize.py): writes outside the workspace fail, shared files
# are copy-on-write, strict mode forbids reads of project files that bypass the workspace; logs/ledger.jsonl lists every input read.
set -euo pipefail
P2=$(cd "$(dirname "$0")" && pwd); REPO=${PRC_ROOT:-$(dirname "$P2")}; PERIOD=${1:?usage: run_pipeline.sh <period> [step ...]}; shift   # repo root: $PRC_ROOT if set, else the parent of phase2/
set -a; source "$P2/periods/$PERIOD.env"; set +a
FORM=${FORM:-${CHAIN_FORM:-px}}; SFX=$([ "$FORM" = px ] && echo "" || echo ".$FORM")
CFORM=$([ "$FORM" = v99 ] && echo v87 || echo "$FORM"); CSFX=$([ "$CFORM" = px ] && echo "" || echo ".$CFORM")   # v99 runs the v87 chains (same steps, same markers)
NEWSTAND_DELTA=${NEWSTAND_DELTA:-$([ "$FORM" = v87 ] && echo 1 || echo 0)}; E1S=${E1S:-0}; WIDELEAF=${WIDELEAF:-0}; PSEUDOLABEL=${PSEUDOLABEL:-0}; F13=${F13:-1}; F4=${F4:-1}; FOLD_LINK=${FOLD_LINK:-1}
WS=$P2/work/${WS_NAME:-$PERIOD}; PY=$REPO/.venv/bin/python; NT=${NT:-6}
CHAINS="fold_dTK fold_Kq fold_NNA3 prod_dTK prod_Kq prod_NNA3"
case $FORM in t2x) CHAINS="p13stk_dTK p13stk_Kq p13stk_NNA3 $CHAINS";; v87) CHAINS="pre_v87 p13stk_dTK p13stk_Kq p13stk_NNA3 $CHAINS";;
  v99) CHAINS="pre_v87 p13stk_dTK p13stk_Kq p13stk_NNA3 $CHAINS v99prep"; [ "$F13" = 1 ] && CHAINS="$CHAINS kumo_f13"; [ "$F4" = 1 ] && CHAINS="$CHAINS kumo_f4"
       [ "$WIDELEAF" = 1 ] && CHAINS="$CHAINS xb_fold_Kq xb_fold_NNA3 xb_prod_Kq xb_prod_NNA3"; CHAINS="$CHAINS iso_L"
       [ "$PSEUDOLABEL" = 1 ] && { [ "$MODE" = new ] && CHAINS="$CHAINS xs_pseudo m_pseudolabel_pl3p88"; CHAINS="$CHAINS xs_fold_Kq xs_fold_NNA3 xs_prod_Kq xs_prod_NNA3 iso_XS"; }
       [ "$WIDELEAF" = 1 ] && CHAINS="$CHAINS iso_XB";;
esac
# member families: the optional wideleaf_* / pseudolabel_* families of manifest/members.tsv run only with their flag (PSEUDOLABEL's member needs the first pass, see CHAINS)
MEMBERS=$(awk -F'\t' '!/^#/ && NF > 5 && $1 !~ /^(wideleaf|pseudolabel)_/ {printf "m_%s ", $1}' "$P2/manifest/members.tsv"); [ "$WIDELEAF" = 1 ] && MEMBERS="${MEMBERS}m_wideleaf_a2p "
if [ "$MODE" = new ]; then DEF="ws np_side ${MEMBERS}np_derive pre $CHAINS post compare"
else DEF="ws pre $CHAINS post compare"; fi
STEPS=${*:-$DEF}
export P2_REPO=$REPO P2_ROOT=$WS PRC_ROOT=$WS P2_LOG=$WS/logs/ledger.jsonl P2_STRICT=${P2_STRICT:-1} PYTHONPATH=$P2/lib/guard daystate_root=$WS daystate_dir=$WS/scratchpad/xl
export POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage P2_NRANK=${NRANK:-344841}
log(){ echo "[$(date +%T)] $*"; }
run(){ local s=$1; shift; [ -f "$WS/logs/$s.done" ] && { log "skip $s (done)"; return 0; }
       log "step $s"; ( cd "$WS" && "$@" ) > "$WS/logs/$s.log" 2>&1 || { log "FAILED $s (see $WS/logs/$s.log)"; tail -20 "$WS/logs/$s.log"; exit 1; }
       # errexit is off inside the ( ... ) || list, so a failing command in the middle of a step function does not stop it: an uncaught
       # Python exception anywhere in the step log fails the step too
       grep -q "^Traceback (most recent call last)" "$WS/logs/$s.log" && { log "FAILED $s (Python traceback in $WS/logs/$s.log)"; grep -A3 "^Traceback" "$WS/logs/$s.log" | tail -20; exit 1; }
       touch "$WS/logs/$s.done"; log "done $s"; }
D=scratchpad/xof; Q=scratchpad/xpr; S=submissions; ST=stages
EXTRA_BASE=lnk,xlk,xdk,xapt; [ "${PRU:-1}" = 1 ] && EXTRA_ALL=$EXTRA_BASE,xpr || EXTRA_ALL=$EXTRA_BASE
CH=$([ "${PRU:-1}" = 1 ] && echo PX || echo PB)      # chain name prefix (PX = with the PRU daily meta inputs, as v71-v76)

step_ws(){ env -u PYTHONPATH -u P2_ROOT -u P2_STRICT $PY "$P2/p2ws.py" "${WS_NAME:-$PERIOD}" --force ${FEATURES:+--features "$FEATURES"} ${RAW_RANK:+--raw-rank "$RAW_RANK"}
           mkdir -p "$WS/logs"
           if [ "$MODE" = repro ]; then ln -sfn "$(readlink -f "$REPO/scratchpad/xl/frame.parquet")" "$WS/scratchpad/xl/frame.parquet"; fi
           # X4 fold frame: 2025 fold rows with their ranking-mix weights (scripts/postrule_fold_frame.py, make constants); a fitted 2025 artefact, read-only
           [ -e "$WS/scratchpad/x4/X4_fold.parquet" ] || ln -sfn "$(readlink -f "$REPO/scratchpad/x4/X4_fold.parquet")" "$WS/scratchpad/x4/X4_fold.parquet"
           if [ "$MODE" = repro ] && [ -d "$REPO/scratchpad/xau2" ]; then   # NEWSTAND_DELTA new-stand delta members + ids (fitted artefacts)
             for f in "$REPO"/scratchpad/xau2/*.parquet; do [ -e "$WS/scratchpad/xau2/$(basename $f)" ] || ln -s "$(readlink -f $f)" "$WS/scratchpad/xau2/$(basename $f)"; done; fi; }

step_pre(){
  $PY "$P2/steps/p2_unmatched.py"
  $PY "$P2/steps/p2_slot.py" --out $ST/unmatched_rule.parquet --unmatched U      # unmatched rows by rule, C1 placeholder on matched rows
  # fallback_base (its values are read on the xba fallback rows only): ADS-B link-flag build: fallback_assemble.py, members runway_queue_v47lq x6 + CATV17x x14 + NNRA2, meta OOF runway_queue_ra3q/RAC2/NNA2 + q + lnk
  rm -f $ST/fallback_lnk_matched.parquet
  env NT=4 EXTRA=lnk PLG=runway_queue_v47lq NLGB=6 PCT=CATV17x NCAT=14 PNN=NNRA2 OL=runway_queue_ra3q OC=RAC2 ON=NNA2 QOOF=fallback_regpl BASE=$ST/unmatched_rule.parquet TAG=lnk \
      $PY scripts/fallback_assemble.py
  $PY "$P2/steps/p2_slot.py" --out $ST/fallback_base.parquet --unmatched U --matched $ST/fallback_lnk_matched.parquet
  # unm_resid unmatched-row model: D (Huber LightGBM on 2025 unmatched rows) + resb stacker, LTFM kept at the baseline (declared scope)
  env NT=4 VARS=HUB $PY scratchpad/xur/unm_resid_d.py
  env NT=4 APT_MEMBER=LTFM DV=HUB STK=resb YCLIP=14400 CLK=0 SHR=0.5 GCL=900 $PY scratchpad/xur/unm_resid_s.py
  $PY "$P2/steps/p2_unm_resid_apply.py" xLTFM_HUB_resb_c14400_k0_s0.5_g900
  $PY "$P2/steps/p2_slot.py" --out $ST/unmatched_resid.parquet --unmatched U2
  step_p13prep
}

step_p13prep(){  # fold-side XL state columns scratchpad/p13/prep_<fold>.parquet (chain_prep.py, defaults OL=queue_eb_ra3qa OC=meta_oof_h_ctl = hist. job 250913985);
                  # read by chain_assemble._xdv_state in every six-seed production step (missing file -> FileNotFoundError in prod_*)
  $PY scripts/chain_prep.py
}
step_pre_v87(){  # unmatched rows of the v82-v87 line + the (B) day-sum proxy (needs step pre)
  $PY scripts/unm_link_characterise.py                   # frame + IATA->ICAO callsign map from matched rows (label-free)
  $PY scripts/unm_link.py 4                 # callsign link of unmatched departures to adsb.lol traces (take-off within 60 s)
  $PY scripts/unm_link_lever.py A1 V3            # A1: LOMO residual on the link clocks (k .5, clip 600); V3: k 1, clip 1200 (same g)
  $PY "$P2/steps/p2_unm_link_apply.py" A1 UA1    # v82 unmatched rows
  $PY "$P2/steps/p2_unm_link_apply.py" V3 U3     # v84..v87 unmatched rows
  $PY "$P2/steps/p2_slot.py" --out $ST/unmatched_link_a1.parquet --unmatched UA1
  $PY scripts/calib_unmatched_proxy.py scratchpad/xau3/um25_A1.npz    # E3: the 2025 day-sum proxy carries the A1 values like the 2026 rows
}
step_p13stk(){  # T2x fold stack + C4/tree D (chain_stk.sh KB<C>, xpr form) = the fold TD shared by the t2x and v87 forms
  env pru=$PRU_FLAG NT=$NT bash scripts/chain_stk.sh KB$1
}
chain_env(){   # environment of the px chains; C in dTK | Kq | NNA3
  local C=$1; export oof_build_folds=6 NT OMP_NUM_THREADS=$NT
  export PLG=queue_eb_v47lqa NLGB=6 PCT=CATV17x NCAT=14 PNN=NNRA2 OL=queue_eb_ra3qa OC=meta_oof_h_ctl ON=NNA2 QOOF=fallback_regpl PXLK=lgb_adsb_linked_a NXLK=3 CLOCK_OOF=lgb_adsb_linked_a METAFIT=pairs \
    BASE=$ST/fallback_base.parquet FALLBACK=data/external/xba_fallback_ids.parquet
  export EXTRA=$EXTRA_ALL DARK_OOF=lgb_noclock_nc PXDK=lgb_noclock_nc NXDK=3 APT_MEMBER=lgb_airport_ltfm:LTFM,lgb_airport_lfpg:LFPG PXAPT=lgb_airport_ltfm:LTFM:3,lgb_airport_lfpg:LFPG:3
  case $C in dTK) export PLG=queue_decomp_ebdt NLGB=3 OL=queue_eb_dt_dt;; Kq) export PCT=runway_queue_catv17xq NCAT=6 OC=runway_queue_catq;; NNA3) export PNN=meta_oof_realmlp_nnra3 ON=meta_oof_realmlp_nna3;; esac
}
PRU_FLAG=$([ "${PRU:-1}" = 1 ] && echo 1 || echo 0); XS=$([ "${PRU:-1}" = 1 ] && echo x || echo "")
step_fold(){    # fold side of one chain: stage fits on the 2025 month folds (member OOF only; independent of the period's ranking rows)
  if [ "$CFORM" = t2x ]; then   # v78 form (P13 T2x = K(b) + xpr + A1): scratchpad/p13/j_T2x_<C>.sh part 2 (chain_stk = step p13stk_<C>)
    env FORM=PC A1=1 NT=$NT bash scripts/chain_stage.sh scratchpad/p13/P13_KB$1${XS}TD KB$1${XS}ca; return; fi
  if [ "$CFORM" = v87 ]; then   # v87 and v99 forms: CLOCKFOLD1 fold stage on the T2x TD (adsb5 + adsb6 + adsbL clocks, A1; as scripts/chain_jobs/clockstage/j_{L,Kq,NNA3}.sh)
    local L=L$1; mkdir -p scratchpad/xau1 scratchpad/xlv
    env clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=${NT_V87:-4} bash scripts/clockstage_fold.sh scratchpad/p13/P13_KB$1${XS}TD $L; return; fi
  local C=$1; chain_env $C; local N=$CH$C XO="daystate_root=$WS daystate_oofdirs=$WS/$D assembler_fl=$OL assembler_fc=$OC"
  # ---- fold side: stage fits on the 2025 month folds (6 folds)
  oof_build_src=scripts/pru_stack.py NAME=$N $PY scripts/stage12_mkstk.py
  env $XO $PY scripts/stage12_run.py scripts/combo_treed_apply.py --target none --pre none --out none --fit-oof oof_build_${N}STK --gate --dry --save-oof $D/oof_build_${N}TD
  env adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600 $PY scripts/adsb5_stage_run.py scripts/clock_fit_tuned.py $D/oof_build_${N}TD $D/oof_build_${N}c --ns 6 --nt $NT --objective huber --jit 25 --ncopy 2 --npz $D/oof_build_${N}c_HuJ.npz
  env adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600 leave_stage_noa1=1 $PY scripts/adsb5_stage_run.py scripts/leave_fit_tuned.py $D/oof_build_${N}c_HuJ.npz --hnpz $D/oof_build_${N}c_HuJ.npz --tag oof_build_${N}c --ns 3 --nt $NT
  $PY scripts/stage12_a0l.py $D/oof_build_${N}c_HuJ.npz oof_build_${N}c $D/oof_build_${N}c_HA0L
}
step_prod(){    # production side of one chain on the period's ranking rows (needs step pre and fold_<C>)
  if [ "$CFORM" = t2x ]; then   # v78 form: fallback rows from the member (FBMODE=member), not from stages/fallback_base
    rm -f scratchpad/p13/PR_KB$1${XS}_member_{pre,C4,C4treeD,C4treeRAW}.parquet
    env pru=$PRU_FLAG FORM=PC A1=1 FBMODE=member NT=${NT_T2X:-4} bash scripts/chain_prod.sh KB$1 KB$1${XS}ca; return; fi
  if [ "$CFORM" = v87 ]; then   # v87 form (and v99): SIXSEED AM pre/C4/tree (6-seed members, 6 meta seeds) + CLOCKFOLD1 production clock stage + L (6 seeds)
    rm -f scratchpad/xau1/PR_KB$1${XS}_AMXAU1_{pre,C4,C4treeD,C4treeRAW}.parquet
    env pru=$PRU_FLAG FORM=PC A1=1 FBMODE=member NT=${NT_V87:-4} ARM=AMXAU1 SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1 bash "$P2/steps/p2_bundle_prod.sh" KB$1 L$1; return; fi
  local C=$1; chain_env $C; local N=$CH$C XO="daystate_root=$WS daystate_oofdirs=$WS/$D assembler_fl=$OL assembler_fc=$OC"
  # ---- production on the period's ranking rows
  local PRE=$Q/${CH}_${C}_pre.parquet C4=$Q/${CH}_${C}_C4.parquet TD=$Q/${CH}_${C}_C4treeD.parquet
  rm -f $PRE $C4 $TD $Q/${CH}_${C}_C4treeRAW.parquet
  env TAG=${CH}_$C ROUND=rint OUT=$PRE $PY scripts/pru_assemble.py
  env $XO $PY scripts/stage12_run.py scripts/combo_c4_apply.py --target $PRE --out $C4 --fit-oof oof_build_${N}STK \
    --feats PR5,PR15,PRT5,PRT15,PRX5,PRX15,PAX15,PRT15u,PRX5u,PRX15u,PAX15u --demean day --pscale --lgb $PLG --nlgb $NLGB --cat $PCT --ncat $NCAT --v53 $ST/unmatched_rule.parquet
  env $XO $PY scripts/stage12_run.py scripts/combo_treed_apply.py --target $C4 --pre $PRE --fit-oof oof_build_${N}STK --lgb $PLG --nlgb $NLGB --cat $PCT --ncat $NCAT \
    --out $Q/${CH}_${C}_C4treeRAW.parquet --out-d $TD
  $PY scripts/stage12_jitter.py $D/oof_build_${N}TD 25 2
  env adsb5_clock_adsb5=1 clock_tune_cleaves=255 clock_tune_crounds=600 $PY scripts/adsb5_stage_run.py scripts/stage12_clock_apply.py --target $TD --fit-oof $D/oof_build_${N}TD,$D/oof_build_${N}TDj1,$D/oof_build_${N}TDj2 \
    --out $Q/${CH}_${C}_Hu6.parquet --fset clkonly --shrink 1.0 --clip 300 --seeds 6 --wmode sqrt --objective huber --alpha 600 --nt $NT --v53 $ST/unmatched_rule.parquet
  $PY scripts/adsb5_stage_run.py scripts/leave_apply.py --target $Q/${CH}_${C}_Hu6.parquet --pre $TD --npz $D/oof_build_${N}c_HuJ.npz --gnpz $D/oof_build_${N}c_HuJ.npz \
    --out $Q/${CH}_${C}_HA0L.parquet --lirf-excl --arm L --seeds 3 --nt $NT
}

step_post(){
  [ "$FORM" = v99 ] && { step_post_v99; return; }
  [ "$FORM" != px ] && { step_post_line; return; }
  local W=0.3212,0.3065,0.3723 POOL=$Q/pool.parquet
  # fixed-weight pool (P12 w5, LOFO-simplex weights fixed on 2025 folds); the two zero-weight slots are kept
  # (filled with chain files) so the weighted mean is computed with the identical 5-vector dot product
  env W=0.0,0.3212,0.3065,0.0,0.3722 $PY scripts/pool_mix.py $ST/unmatched_rule.parquet $POOL $Q/${CH}_dTK_HA0L.parquet $Q/${CH}_dTK_HA0L.parquet \
      $Q/${CH}_Kq_HA0L.parquet $Q/${CH}_dTK_HA0L.parquet $Q/${CH}_NNA3_HA0L.parquet
  local B=$POOL FP=$D/oof_build_${CH}
  [ "${CHAIN_FORM:-px}" = t2x ] && FP=scratchpad/p13/P13_KB
  local F1=${FP}dTK${XS}c_HA0L F2=${FP}Kq${XS}c_HA0L F3=${FP}NNA3${XS}c_HA0L
  [ "${CHAIN_FORM:-px}" = t2x ] && { F1=${FP}dTK${XS}ca_HA0L; F2=${FP}Kq${XS}ca_HA0L; F3=${FP}NNA3${XS}ca_HA0L; }
  [ "${CHAIN_FORM:-px}" = px ] && [ "${PRU:-1}" = 1 ] && { F1=$D/oof_build_${CH}dTKc_HA0L; F2=$D/oof_build_${CH}Kqc_HA0L; F3=$D/oof_build_${CH}NNA3c_HA0L; }
  [ "${CHAIN_FORM:-px}" = px ] && [ "${PRU:-1}" != 1 ] && { F1=$D/oof_build_${CH}dTKc_HA0L; F2=$D/oof_build_${CH}Kqc_HA0L; F3=$D/oof_build_${CH}NNA3c_HA0L; }
  if [ "${PRU:-1}" = 1 ]; then
    # (B) soft daily calibration to the EUROCONTROL/PRU daily pre-departure delay: fold fit on the pool OOF, unmatched rows of
    # the day sums: unm_resid OOF (2025) / the U2 rule values (period)
    $PY scripts/pru_daily_calib.py --pool $F1,$F2,$F3 --w $W --out $Q/pru_pxx \
      --um25 scratchpad/xur/S_xLTFM_HUB_resb_c14400_k0_s0.5_g900.npz --prod $POOL --prod-out $Q/pool_B.parquet --prod-arm B --prod-um prod --um-src $ST/unmatched_resid.parquet
    B=$Q/pool_B.parquet
  fi
  $PY "$P2/steps/p2_slot.py" --out $Q/xprBX.parquet --unmatched U2 --matched $B        # matched rows from the chain, unmatched from U2
  # post-chain rules: parameters fitted on 2025 only = the chain's final fold vectors + the X4 fold frame's row weights (scratchpad/x4)
  local FOLDV=$Q/pru_pxxb
  if [ "${PRU:-1}" != 1 ]; then $PY scripts/stage12_mix.py $D/oof_build_${CH}POOL $W $F1 $F2 $F3; FOLDV=$D/oof_build_${CH}POOL; fi
  if [ -n "${POSTRULES_PARAMS:-}" ]; then cp "$P2/$POSTRULES_PARAMS" scratchpad/p2/postrules.json     # fixed per-airport arms (2025 fit)
  else $PY "$P2/steps/p2_postrules.py" fit --fold $FOLDV --x4 scratchpad/x4/X4_fold.parquet --out scratchpad/p2/postrules.json; fi
  $PY "$P2/steps/p2_postrules.py" apply --in $Q/xprBX.parquet --out $Q/xprBX_xsx.parquet --params scratchpad/p2/postrules.json --only xsx
  $PY "$P2/steps/p2_postrules.py" apply --in $Q/xprBX_xsx.parquet --out $Q/xprBX_xsx_xpk.parquet --params scratchpad/p2/postrules.json --only xpk
  $PY "$P2/steps/p2_exposed_scenarios.py" --in $Q/xprBX_xsx_xpk.parquet --out $S/final.parquet
  $PY "$P2/steps/p2_check.py" validate $S/final.parquet
}

step_post_line(){   # post for the t2x (v78) and v87 forms: the clock-stage line build, by rule
  local W=0.3212,0.3065,0.3723 POOL=$Q/pool$SFX.parquet B=$Q/pool$SFX.B.parquet P=$Q/post$SFX
  local C1 C2 C3 F UM25 UMSRC UNM
  if [ "$FORM" = t2x ]; then
    C1=scratchpad/p13/PR_KBdTK${XS}ca_member_HA0L.parquet; C2=scratchpad/p13/PR_KBKq${XS}ca_member_HA0L.parquet; C3=scratchpad/p13/PR_KBNNA3${XS}ca_member_HA0L.parquet
    F=scratchpad/p13/P13_KBdTK${XS}ca_HA0L,scratchpad/p13/P13_KBKq${XS}ca_HA0L,scratchpad/p13/P13_KBNNA3${XS}ca_HA0L
    UM25=scratchpad/xur/S_xLTFM_HUB_resb_c14400_k0_s0.5_g900.npz; UMSRC=$ST/unmatched_resid.parquet; UNM=U2
  else
    C1=scratchpad/xau1/PR_LdTK_AMXAU1_HA0L.parquet; C2=scratchpad/xau1/PR_LKq_AMXAU1_HA0L.parquet; C3=scratchpad/xau1/PR_LNNA3_AMXAU1_HA0L.parquet
    F=scratchpad/xau1/clockfold1_ldtk_ha0l,scratchpad/xau1/clockfold1_lkq_ha0l,scratchpad/xau1/clockfold1_lnna3_ha0l
    UM25=scratchpad/xau3/um25_A1.npz; UMSRC=$ST/unmatched_link_a1.parquet; UNM=U3
  fi
  env W=0.0,0.3212,0.3065,0.0,0.3722 $PY scripts/pool_mix.py $ST/unmatched_rule.parquet $POOL $C1 $C1 $C2 $C1 $C3
  if [ "${PRU:-1}" = 1 ]; then
    $PY scripts/pru_daily_calib.py --pool $F --w $W --out $Q/pru$SFX --um25 $UM25 --prod $POOL --prod-out $B --prod-arm B --prod-um prod --um-src $UMSRC
  else cp $POOL $B; fi
  $PY "$P2/steps/p2_slot.py" --out $P.stitch.parquet --unmatched $UNM --matched $B
  cp "$P2/${POSTRULES_PARAMS:-params/postrules_fixed.json}" scratchpad/p2/postrules$SFX.json    # fixed per-airport arms (2025 fit)
  $PY "$P2/steps/p2_postrules.py" apply --in $P.stitch.parquet --out $P.xsx.parquet --params scratchpad/p2/postrules$SFX.json --only xsx
  $PY "$P2/steps/p2_postrules.py" apply --in $P.xsx.parquet --out $P.xpk.parquet --params scratchpad/p2/postrules$SFX.json --only xpk
  local LAST=$P.xpk.parquet
  if [ "$NEWSTAND_DELTA" = 1 ]; then         # NEWSTAND_DELTA new-stand neighbour delta on matched rows of stands unseen in 2025 (default 1 for FORM=v87, 0 otherwise; off in the v88 post line and in v99)
    $PY scripts/newstand_apply.py $P.xpk.parquet scratchpad/xau2/features_v9_prodN_ids.parquet $P.xau2.parquet scratchpad/xau2; LAST=$P.xau2.parquet; fi
  $PY "$P2/steps/p2_exposed_scenarios.py" --in $LAST --out $S/final$SFX.parquet --ledger scratchpad/p2/xhr_ledger$SFX.parquet
  $PY "$P2/steps/p2_check.py" validate $S/final$SFX.parquet
}

# ---------------- FORM=v99 (STAGE_COMPARE_V99): the v88 -> v99 levers as rules, each on the previous workspace output
KPY=${KUMO_PY:-$REPO/.venv-kumo/bin/python}; NTK=${NT_KUMO:-4}
kumo_env(){   # Kumo Tabular ICL on CPU (safe mode, as the uploaded v96-v99: 4 threads); weights read through the workspace link artifacts/xkumo
  export DEV=cpu PRC_THREADS=$NTK NT=$NTK OMP_NUM_THREADS=$NTK HF_HOME=$WS/artifacts/xkumo/hf HF_HUB_OFFLINE=1 P2_MODE=new
  export P2_ALLOW="$(readlink -f "$WS/artifacts/xkumo"):${P2_ALLOW:-}"
  export P2_STRICT=0; }   # huggingface_hub resolves the cache link to its real path: strict mode would refuse the weight files (still logged in the ledger)
step_kumo_f13(){   # Kumo F1 (UL: linked unm_other, V3 form) and F3 (UU: unlinked unm_other); LOMO on the 2025 unm_other rows, ranking rows from all 12 months
  kumo_env; export OUT=scratchpad/xtfm2
  $KPY scripts/kumo_icl_unmatched.py F1; $KPY scripts/kumo_icl_unmatched.py F3; ls -la scratchpad/xtfm2/ARM_F1.npz scratchpad/xtfm2/ARM_F3.npz
}
step_kumo_f4(){    # Kumo F4 (unm_LIRF, frozen blend's own p / taxi as inputs); the arm's incumbent rows are read from the U3 slot (unm_LIRF = the frozen
                   # blend rule there and in every later file; BUILD in post asserts it)
  kumo_env; export OUT=scratchpad/xtfm2/frz kumo_icl_lirf_frz_base=$ST/unmatched_link_a1.parquet kumo_icl_lirf_frz_base_md5=
  $KPY scripts/kumo_icl_lirf.py F4; ls -la scratchpad/xtfm2/frz/ARM_F4.npz
}
step_xb_fold(){    # WIDELEAF fold side of chain $1: T2x stack with the xtd input -> CLOCKFOLD1 clock stage; FOLD_LINK=1: checks the 2025-fitted files that step v99prep links
  if [ "$FOLD_LINK" = 1 ]; then ls scratchpad/xbase/P13_XB$1xTD_janjul.npy scratchpad/xau1/clockfold1_xb$1_HuJ.npz scratchpad/xau1/clockfold1_xb$1_HA0L_janjul.npy; echo "fold side linked (FOLD_LINK=1)"; return; fi
  env pru=1 NT=$NT bash scripts/wideleaf_stk.sh XB$1
  env clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=${NT_V87:-4} bash scripts/clockstage_fold.sh scratchpad/xbase/P13_XB$1xTD XB$1
}
step_xb_prod(){    # WIDELEAF production chain $1: members wideleaf_a2p_s0-2 as the xtd input, CLOCKFOLD1 form (AM seeds, MS6, adsb6 + adsbL)
  rm -f scratchpad/xbase/PR_XB$1x_XBAU1_{pre,C4,C4treeD,C4treeRAW}.parquet
  env pru=$PRU_FLAG FORM=PC A1=1 FBMODE=member NT=${NT_V87:-4} ARM=WIDELEAF_CLOCKFOLD_ARM ARMIN=XB SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1 bash scripts/wideleaf_bundle_prod.sh XB$1 XB$1
}
step_xs_pseudo(){  # PSEUDOLABEL pass 2 input (MODE=new): matched ranking rows pseudo-labelled by the period's OWN first-pass prediction iso_L (our output, no label)
  env pseudolabel_teacher=$S/v99_iso_L.parquet pseudolabel_prod_out=scratchpad/xsyn/pseudo_prod_chain88.parquet $PY scripts/pseudo_labels.py prod
}
step_xs_fold(){    # PSEUDOLABEL fold side of chain $1 (pseudo_fold.sh): needs data/oof/pseudolabel_pl3_<fold> (2025); FOLD_LINK=1: linked
  if [ "$FOLD_LINK" = 1 ]; then ls scratchpad/xsyn/P13_XSPL3$1xTD_janjul.npy scratchpad/xau1/clockfold1_xspl3$1_HuJ.npz scratchpad/xau1/clockfold1_xspl3$1_HA0L_janjul.npy; echo "fold side linked (FOLD_LINK=1)"; return; fi
  env pru=1 NT=$NT bash scripts/pseudo_stk.sh PL3 $1
  env clockfold1_adsbl=1 pushstart_adsb6=1 FORM=PC A1=1 NT=${NT_V87:-4} bash scripts/clockstage_fold.sh scratchpad/xsyn/P13_XSPL3$1xTD PSEUDOLABEL_CHAIN$1
}
step_xs_prod(){    # PSEUDOLABEL production chain $1: L member = pseudolabel_pl3p88 s0, s1
  [ "$MODE" = new ] && P2_STRICT=0 $PY "$P2/steps/p2_alias.py"            # seeds of pseudolabel_pl3p88 not retrained -> seed 0 (the member is trained after np_derive)
  rm -f scratchpad/xsyn/PR_XSPL3$1x_XSAU1_{pre,C4,C4treeD,C4treeRAW}.parquet
  env pru=$PRU_FLAG FORM=PC A1=1 FBMODE=member NT=${NT_V87:-4} ARM=PSEUDOLABEL_CLOCKFOLD_ARM ARMIN=XS SEED6=1 MS6=1 pushstart_adsb6=1 clockfold1_adsbl=1 bash scripts/pseudo_bundle_prod.sh PSEUDOLABEL_CHAIN$1 PSEUDOLABEL_CHAIN$1
}
post_iso(){        # one isolate of the v88 line = step_post_line (v87 form) without NEWSTAND_DELTA:  $1 tag, $2/$3/$4 production HA0L of dTK/Kq/NNA3, $5 fold HA0L prefixes
  local T=$1 POOL=$Q/pool.v99$1.parquet B=$Q/pool.v99$1.B.parquet P=$Q/post.v99$1
  env W=0.0,0.3212,0.3065,0.0,0.3722 $PY scripts/pool_mix.py $ST/unmatched_rule.parquet $POOL $2 $2 $3 $2 $4
  if [ "${PRU:-1}" = 1 ]; then
    $PY scripts/pru_daily_calib.py --pool $5 --w 0.3212,0.3065,0.3723 --out $Q/pru.v99$1 --um25 scratchpad/xau3/um25_A1.npz --prod $POOL --prod-out $B --prod-arm B --prod-um prod \
      --um-src $ST/unmatched_link_a1.parquet
  else cp $POOL $B; fi
  $PY "$P2/steps/p2_slot.py" --out $P.stitch.parquet --unmatched U3 --matched $B
  cp "$P2/${POSTRULES_PARAMS:-params/postrules_fixed.json}" scratchpad/p2/postrules.v99.json
  $PY "$P2/steps/p2_postrules.py" apply --in $P.stitch.parquet --out $P.xsx.parquet --params scratchpad/p2/postrules.v99.json --only xsx
  $PY "$P2/steps/p2_postrules.py" apply --in $P.xsx.parquet --out $P.xpk.parquet --params scratchpad/p2/postrules.v99.json --only xpk
  local LAST=$P.xpk.parquet
  if [ "$NEWSTAND_DELTA" = 1 ]; then $PY scripts/newstand_apply.py $P.xpk.parquet scratchpad/xau2/features_v9_prodN_ids.parquet $P.xau2.parquet scratchpad/xau2; LAST=$P.xau2.parquet; fi
  $PY "$P2/steps/p2_exposed_scenarios.py" --in $LAST --out $S/v99_iso_$T.parquet --ledger scratchpad/p2/xhr_ledger.v99$T.parquet
  $PY "$P2/steps/p2_check.py" validate $S/v99_iso_$T.parquet
}
AX=scratchpad/xau1
step_iso_L(){  post_iso L  $AX/PR_LdTK_AMXAU1_HA0L.parquet $AX/PR_LKq_AMXAU1_HA0L.parquet $AX/PR_LNNA3_AMXAU1_HA0L.parquet $AX/clockfold1_ldtk_ha0l,$AX/clockfold1_lkq_ha0l,$AX/clockfold1_lnna3_ha0l; }
step_iso_XB(){ post_iso XB $AX/PR_LdTK_AMXAU1_HA0L.parquet scratchpad/xbase/PR_XBKq_XBAU1_HA0L.parquet scratchpad/xbase/PR_XBNNA3_XBAU1_HA0L.parquet \
                 $AX/clockfold1_ldtk_ha0l,$AX/clockfold1_xbkq_ha0l,$AX/clockfold1_xbnna3_ha0l; }
step_iso_XS(){ post_iso XS $AX/PR_LdTK_AMXAU1_HA0L.parquet scratchpad/xsyn/PR_XSPL3Kq_XSAU1_HA0L.parquet scratchpad/xsyn/PR_XSPL3NNA3_XSAU1_HA0L.parquet \
                 $AX/clockfold1_ldtk_ha0l,$AX/clockfold1_xspl3kq_ha0l,$AX/clockfold1_xspl3nna3_ha0l; }
step_post_v99(){   # compose (v94 rule) -> E1s -> Kumo F1/F3 -> Kumo F4 -> final.v99; every intermediate in scratchpad/xpr/v99_*.parquet
  local CUR=$Q/v99_comp.parquet ISOS=""
  [ "$WIDELEAF" = 1 ] && ISOS="$ISOS $S/v99_iso_XB.parquet"; [ "$PSEUDOLABEL" = 1 ] && ISOS="$ISOS $S/v99_iso_XS.parquet"
  if [ -n "$ISOS" ]; then $PY "$P2/steps/p2_compose.py" --base $S/v99_iso_L.parquet --out $CUR $ISOS; else cp $S/v99_iso_L.parquet $CUR; fi
  if [ "$E1S" = 1 ]; then $PY "$P2/steps/p2_e1s.py" --in $CUR --out $Q/v99_e1s.parquet; CUR=$Q/v99_e1s.parquet; fi
  if [ "$F13" = 1 ]; then env kumo_icl_lirf_base=$CUR kumo_icl_lirf_base_md5= P2_MODE=new $PY scripts/kumo_icl_build.py $Q/v99_f13.parquet UL,UU; CUR=$Q/v99_f13.parquet; fi
  if [ "$F4" = 1 ]; then
    rm -f $Q/v99_f4F4.parquet
    env kumo_icl_lirf_frz_base=$CUR kumo_icl_lirf_frz_base_md5= kumo_icl_lirf_frz_cand="$Q/v99_f4{tag}.parquet" P2_MODE=new OUT=scratchpad/xtfm2/frz $PY scripts/kumo_icl_lirf.py SUMMARY
    env kumo_icl_lirf_frz_base=$CUR kumo_icl_lirf_frz_base_md5= kumo_icl_lirf_frz_cand="$Q/v99_f4{tag}.parquet" P2_MODE=new OUT=scratchpad/xtfm2/frz $PY scripts/kumo_icl_lirf.py BUILD
    CUR=$Q/v99_f4F4.parquet; fi
  cp $CUR $S/final.v99.parquet
  echo "v99 form: NEWSTAND_DELTA=$NEWSTAND_DELTA WIDELEAF=$WIDELEAF PSEUDOLABEL=$PSEUDOLABEL E1S=$E1S F13=$F13 F4=$F4 FOLD_LINK=$FOLD_LINK -> $S/final.v99.parquet"
  $PY "$P2/steps/p2_check.py" validate $S/final.v99.parquet
}

# ---------------- MODE=new: period-dependent inputs (features are given by FEATURES / p2_mkdry.py or built by np_features)
step_np_features(){   # a real new period: official ranking file -> features (build_features.py --rank) + the per-batch dreg table
  $PY scripts/build_features.py --rank data/raw/ranking.parquet --out data/features_v9.parquet
}
step_np_side(){       # label-free per-batch side tables that depend on WHICH rows are the ranking rows
  $PY scripts/build_dreg_features.py --features data/features_v9.parquet --out data/external/dreg_features.parquet
  $PY scripts/xba_fallback_ids.py
  $PY "$P2/steps/p2_period_inputs.py" --check      # PRU daily / ADS-B / METAR coverage of the period's days (report; fallbacks in docs/REPRODUCE.md)
  env NT=$NT $PY scripts/fallback_classifier.py rank L REGP     # q on the ranking rows (all-2025 fit) -> data/oof/fallback_regpl_rank.npy
  env NT=$NT $PY scripts/queue_regime_prob.py prod           # nested q side table for queue_decomp_ebqq -> data/external/queue_decomp_q_prod_features.parquet
}
step_member(){        # one member family from manifest/members.tsv, seeds $MSEEDS (default: all seeds of the v76 build)
  local FAM=$1 line; line=$(awk -F'\t' -v f="$FAM" '$1 == f' "$P2/manifest/members.tsv"); [ -n "$line" ] || { echo "no member $FAM"; return 2; }
  local rng py cmd; rng=$(echo "$line" | cut -f2); py=$(eval echo "$(echo "$line" | cut -f6)"); cmd=$(echo "$line" | cut -f7)
  local lo=${rng#s}; lo=${lo%-*}; local hi=${rng#*-}; local seeds=${MSEEDS:-$(seq -s' ' $lo $hi)}
  local host; host=$(echo "$line" | cut -f9); echo "member $FAM: production host(s) '$host', this run: $(hostname) $(uname -m) -> a rebuild on another CPU architecture is a new draw of the recipe"
  mkdir -p experiments/XQ
  for S in $seeds; do
    rm -f data/preds/${FAM}_s$S.parquet                      # a prediction linked from the shared data/preds is never reused: every seed is refitted in this workspace
    $py ${cmd//\{S\}/$S}
  done
}
step_np_derive(){     # derived members + dry-run seed aliasing (seeds that were not retrained point at the retrained seed 0)
  # the workspace's data/preds starts as links to the shared predictions; queue_mkdt refuses to overwrite, so drop the links of
  # the derived outputs first (links only: a file written in this workspace is never removed)
  for k in $(seq 0 $(( ${NDT:-1} - 1 ))); do for o in queue_decomp_ebdt sixseed_chain_ebdt6; do [ -L data/preds/${o}_s$k.parquet ] && rm -f data/preds/${o}_s$k.parquet; done; done
  env PA=queue_eb_v47lqa PB=queue_decomp_ebqq PO=queue_decomp_ebdt $PY scripts/queue_mkdt.py ${NDT:-1}
  if [ "$CFORM" = v87 ] || [ "${V87_MEMBERS:-0}" = 1 ]; then       # v87 line: 6-seed dT re-mix and the NNA3 6-seed link family
    env PA=queue_eb_v47lqa PB=queue_decomp_ebqq PO=sixseed_chain_ebdt6 $PY scripts/queue_mkdt.py ${NDT:-1}
    for k in 0 1 2 3 4 5; do [ -e data/preds/meta_oof_realmlp_nnra3_s$k.parquet ] && ! [ -L data/preds/meta_oof_realmlp_nnra3_s$k.parquet ] && { rm -f data/preds/sixseed_chain_nnra3x6_s$k.parquet; ln -s meta_oof_realmlp_nnra3_s$k.parquet data/preds/sixseed_chain_nnra3x6_s$k.parquet; }; done
  fi
  P2_STRICT=0 $PY "$P2/steps/p2_alias.py"
}

step_np_xau2(){       # v87 line, MODE=new: NEWSTAND_DELTA new-stand neighbour mapping + its delta members
  mkdir -p scratchpad/xau2
  $PY scripts/newstand_map.py prod N scratchpad/xau2/features_v9_prodN.parquet          # -> features_v9_prodN.parquet + _ids.parquet
  $PY -c "import polars as pl; pl.read_parquet('data/features_v9.parquet', columns=['MVT_ID_mvt','is_rank','unmatched']).filter((pl.col('is_rank')==1)&(pl.col('unmatched')==1)).select('MVT_ID_mvt').write_parquet('scratchpad/xau2/unm_ids.parquet')"
  for S in ${MSEEDS:-0 1}; do for A in U N; do rm -f data/preds/newstand_delta_p${A}_s$S.parquet; bash scripts/newstand_members.sh $A $S $NT; cp data/preds/newstand_delta_p${A}_s$S.parquet scratchpad/xau2/; done; done
  for A in U N; do [ -e scratchpad/xau2/newstand_delta_p${A}_s1.parquet ] || cp scratchpad/xau2/newstand_delta_p${A}_s0.parquet scratchpad/xau2/newstand_delta_p${A}_s1.parquet; done   # alias seed 1 in a reduced run
}
step_l0check(){      # layer 0: rebuild the features table from the raw files with the current build_features.py and diff it against features_v9
  # (features_v9 predates data/external/enroute_daily.parquet: the rebuild carries 15 extra ert_* columns, reported as "only rebuilt" and not
  #  compared; no member uses the ert group. For a byte-exact file hide enroute_daily from the workspace: rm data/external/enroute_daily.parquet.)
  mkdir -p report; rm -f scratchpad/p2/features_rebuild.parquet
  $PY scripts/build_features.py --rank data/raw/ranking.parquet --out scratchpad/p2/features_rebuild.parquet
  $PY "$P2/steps/p2_features_check.py" scratchpad/p2/features_rebuild.parquet data/features_v9.parquet --json report/l0check.json
}
step_compare(){
  [ -n "${REF:-}" ] || { echo "no REF"; return 0; }
  mkdir -p "$WS/report"
  P2_STRICT=0 $PY "$P2/steps/p2_check.py" compare $S/final$SFX.parquet "$REF" final$SFX --json report/final_vs_ref$SFX.json
}

for s in $STEPS; do
  case $s in
    ws) log "step ws"; step_ws; mkdir -p "$WS/logs"; touch "$WS/logs/ws.done";;
    newperiod) log "newperiod = np_side + m_<family> for every family in manifest/members.tsv + np_derive (run them as separate jobs)";;
    np_features|np_side|np_derive|np_xau2|l0check) run $s step_$s;;
    prep) mkdir -p "$WS/logs"; [ -f "$WS/logs/prep.done" ] || { log "step prep (synthetic period from labelled data: $DRY_ARGS)"
          env -u PYTHONPATH -u P2_ROOT -u P2_STRICT $PY "$P2/steps/p2_mkdry.py" "$PERIOD" $DRY_ARGS > "$WS/logs/prep_mkdry.log" 2>&1 && step_ws >> "$WS/logs/prep_mkdry.log" 2>&1 \
          && touch "$WS/logs/prep.done" "$WS/logs/ws.done" || { log "FAILED prep"; tail -20 "$WS/logs/prep_mkdry.log"; exit 1; }; };;
    score) mkdir -p "$WS/report"; ( cd "$WS" && env -u PYTHONPATH -u P2_ROOT -u P2_STRICT $PY "$P2/steps/p2_score.py" "${WS_NAME:-$PERIOD}" $SCORE_ARGS --file submissions/final$SFX.parquet --tag "$FORM" ) > "$WS/logs/score$SFX.log" 2>&1 && touch "$WS/logs/score$SFX.done"; cat "$WS/logs/score$SFX.log";;
    m_*) run $s step_member ${s#m_};;
    pre) run pre step_pre;;
    pre_v87) run pre_v87 step_pre_v87;;
    p13prep) run p13prep step_p13prep;;
    p13stk_*) run $s step_p13stk ${s#p13stk_};;
    fold_*) run $s$CSFX step_fold ${s#fold_};;
    prod_*) run $s$CSFX step_prod ${s#prod_};;
    chain_*) run fold_${s#chain_}$CSFX step_fold ${s#chain_}; run prod_${s#chain_}$CSFX step_prod ${s#chain_};;
    v99prep) mkdir -p "$WS/logs"; [ -f "$WS/logs/v99prep.done" ] && log "skip v99prep (done)" || { log "step v99prep"
          env -u PYTHONPATH -u P2_ROOT -u P2_STRICT $PY "$P2/steps/p2_v99_prep.py" "$WS" "$REPO" "$MODE" > "$WS/logs/v99prep.log" 2>&1 \
          && ( cd "$WS" && $PY "$P2/steps/p2_unm_cross.py" && $PY scripts/rulebag_blend.py ) >> "$WS/logs/v99prep.log" 2>&1 \
          && { touch "$WS/logs/v99prep.done"; log "done v99prep"; } || { log "FAILED v99prep"; tail -20 "$WS/logs/v99prep.log"; exit 1; }; };;
    kumo_f13|kumo_f4|xs_pseudo|iso_L|iso_XB|iso_XS) run $s step_$s;;
    xb_fold_*|xb_prod_*|xs_fold_*|xs_prod_*) run $s step_${s%_*} ${s##*_};;
    post) run post$SFX step_post;;
    compare) rm -f "$WS/logs/compare$SFX.done"; run compare$SFX step_compare;;
    *) echo "unknown step $s"; exit 2;;
  esac
done
