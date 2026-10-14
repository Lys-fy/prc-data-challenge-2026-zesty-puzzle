#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# NEWSTAND_DELTA: production delta-transfer member. usage: bash scripts/newstand_members.sh U|N SEED [threads]
# U = data/features_v9.parquet (today's production features); N = new-stand neighbour-mapped ranking rows (newstand_map.py prod N)
set -euo pipefail; cd ${PRC_ROOT:-$PWD}; A=$1; S=$2; T=${3:-8}
FEAT=data/features_v9.parquet
if [ "$A" = N ]; then FEAT=scratchpad/xau2/features_v9_prodN.parquet; [ -f $FEAT ] || .venv/bin/python scripts/newstand_map.py prod N $FEAT; fi
DROP=te_fno_,ref_sr_n,ref_s_n,ref_r_n,ref_sh_n,te_op_n,te_cs_n,te_des_n,te2_sr_n,te2_srh_n,te2_opt_n,te2_opr_n,te2_how_n,te2_rhow_n,te2_des_n,te2_tr_n,te2_csr_n,te2_sdes_n,doy
.venv/bin/python scripts/train2.py --name newstand_delta_p${A}_s$S --f32 --strategy mixture --mix-drop fbpru --tail-gate 0.5 --tail-gate-lobt --target resid_aobt3 --es-clean \
  --lr 0.01 --fixed-rounds 2000 --leaves 497 --min-leaf 100 --l2 4.3 --ff 0.63 --cls-leaves 63 --cls-rounds 400 --cls-ff 0.99 --cls-weight stake --stake-power 2.42 --p-power 1.0 \
  --dayshift --dayshift-midnight --features $FEAT --feats base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,nbrx,arrx,durp,dreg,adsb,adsb2,adsb3,xba,xe \
  --drop-cols $DROP --stake-floor 0.5 --seed $S --predict-rank --threads $T --out scratchpad/xau2/results.jsonl
