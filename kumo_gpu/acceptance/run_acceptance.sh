#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# ONE-ENTRY-POINT cluster ACCEPTANCE workload for the Kumo FM engine, ending with arm M stage 2. Nothing is transferred by
# this script; it runs where the bundle (code + data/payload) already is.  usage, from the bundle root:
#   MODE=box   bash acceptance/run_acceptance.sh        # plain multi-GPU SSH box (RunPod / Lambda / a cluster node in an interactive allocation)
#   MODE=slurm bash acceptance/run_acceptance.sh        # login node; fill jobs/config.env first (partition, account, gres); each step = sbatch --wait
#   STEPS=1,2,3 (default 1,2,3,4,5)   NGPU_EXPECT=4   STOP_ON_FAIL=1 (default)   RUN=<scratch dir>   ENVDIR=<venv>   SCALE_MIN=3.0 (speed-up bar; 1-GPU box: NGPU_EXPECT=1 SCALE_MIN=0.8)
#   OFFLINE=1 (default: no GitHub/HF access; sdm + weights from data/payload/vendor)   QMAX=4096   RESUME_KILL_S=45   TIME_STEP=01:00:00 (slurm)
#   LONGBENCH=1 (optional, off: one whole-pool bench unit janjul x LTFM at 100 cols = the first real 250k-context number, d -> 03_longbench.txt)
# Steps and PASS criteria (expected GPU-h on one H100, from the GPU workstation fp16 pilot x 5-8; the box numbers replace them in SUMMARY.md):
#   1 env     setup_env.sh (install, sdm ee6acd402, weights sha256 medium+large, GPU smoke fp32 + fp16) + env_check.py (CUDA, GPUs visible >= NGPU_EXPECT,
#             fp16 matmul, fused SDPA backend, long-key (262k) fused-attention probe incl. GQA with math excluded (peak <= 4 GiB), sdm Triton kernel
#             compile/agreement probe, arch WARN (sm_XY missing, sm_X0 present), payload + the 7 importance files present).                                      PASS: SETUP_OK and ENVCHECK verdict PASS.   ~0.05 GPU-h
#   2 parity  km_engine parity16 = shared20k construction (medium, shared stratified 20k, E 4) in fp16 on janjul x LSZH vs the fp32 GPU workstation file.
#             PASS: RMS diff < 2 s AND corr > 0.9995 (build machine: 0.39 s / 0.999999).                                                           ~0.02 GPU-h
#   3 bench   grid medium/large x 20k/50k x 100/412 columns, one instance each, QMAX 4096 -> $RUN/bench.jsonl.  PASS: 8 BENCH lines, finite rmse,
#             no OOM/guard event.  (GPU workstation fp16 reference ms/row: m20/412 0.80, m20/100 0.23, m50/100 0.24, m50/412 0.85, l20/100 0.26, l20/412 0.84, l50/100 0.30;
#             l50/412 never completed on the build machine.)                                                                                            ~0.1 GPU-h
#   4 scale   the first 4 arm-M units (months39 x LTFM, EHAM, EGLL, LFPG) one per GPU (slurm: array 0-3; box: 4 processes, CUDA_VISIBLE_DEVICES=g, --claim
#             locks). PASS: 4 result files, each exactly once (no collisions); speed-up = (sum of unit secs) / wall >= 3.0; resume: an engine instance
#             (KME8c100 inst 0, janjul LSZH) killed after RESUME_KILL_S = 45 s then re-run completes with rows_predicted_now < query (part files reused).  ~0.1 GPU-h
#   5 armM    the remaining 16 arm-M units (months39 + months410, resume skips step 4's), 4 GPUs; precollect.py -> results/oof/kumo_member_screen_km_{months39,months410}.npy
#             + results/MD5SUMS.  PASS: PRECOLLECT verdict PASS (10 airports per fold, finite, one setting).                                 ~0.3-0.4 GPU-h
# Total ~0.6-1 GPU-h (~15-20 min wall on 4 H100). Outputs: $RUN/acceptance/*.txt, $RUN/acceptance/SUMMARY.md (relayable), $RUN/results/, $RUN/bench.jsonl.
# Bring back: copy $RUN/results to the build machine and collect there (docs/kumo_gpu.md).
set -uo pipefail; B=$(cd "$(dirname "$0")/.." && pwd); cd "$B"
MODE=${MODE:-box}; STEPS=${STEPS:-1,2,3,4,5}; STOP_ON_FAIL=${STOP_ON_FAIL:-1}; export NGPU_EXPECT=${NGPU_EXPECT:-4}
if [ "$MODE" = slurm ]; then source "$B/jobs/config.env"; grep -q "__PARTITION__\|__ACCOUNT__\|__GPUTYPE__" "$B/jobs/config.env" && { echo "fill jobs/config.env first"; exit 2; }; fi
export BUNDLE=$B PAYLOAD=${PAYLOAD:-$B/data/payload} RUN=${RUN:-$B/run} ENVDIR=${ENVDIR:-$B/.venv-kumo}
export HF_HOME=${HF_HOME:-$PAYLOAD/vendor/hf} HF_HUB_OFFLINE=1
export SDM_CHUNK_MEMORY_FRACTION=${SDM_CHUNK_MEMORY_FRACTION:-0.05} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PRC_THREADS=${PRC_THREADS:-4} MEMGUARD_GIB=${MEMGUARD_GIB:-10}
export WITH_LARGE=1 PROFILE=shared20k AMP=fp16 NAME=kumo_member_screen_km
A=$RUN/acceptance; mkdir -p "$A" "$RUN/logs" "$RUN/lists"; PY=$ENVDIR/bin/python; ENG=cluster/km_engine.py; TASK=cluster/km_task.py
APTS="LTFM EHAM EGLL LFPG EDDF LEMD LEBL EDDM LIRF LSZH"; U4=""; UALL=""
for f in months39 months410; do for a in $APTS; do UALL="$UALL member:$f:$a"; done; done; set -- $UALL; U4="$1 $2 $3 $4"
printf '%s\n' $UALL > "$RUN/lists/armM.txt"; printf '%s\n' $U4 > "$RUN/lists/scale4.txt"   # one unit per line, no blank lines (array index = line)
echo "== KUMO_ENGINE acceptance | mode $MODE | steps $STEPS | host $(hostname) | $(date -u) | RUN $RUN" | tee "$A/00_header.txt"
step_on() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
fail() { echo "STEP $1 FAIL: $2" | tee -a "$A/verdicts.txt"; [ "$STOP_ON_FAIL" = 1 ] && { summarize; exit 1; }; }
pass() { echo "STEP $1 PASS: $2" | tee -a "$A/verdicts.txt"; }
summarize() { ("$PY" acceptance/summarize.py "$RUN" 2>/dev/null || python3 acceptance/summarize.py "$RUN") > "$A/SUMMARY.md"; echo "SUMMARY -> $A/SUMMARY.md"; cat "$A/SUMMARY.md"; }
# gpu_run <log> <cmd...>: one GPU, foreground; slurm = sbatch --wait on 1 GPU
gpu_run() { local log=$1; shift; local cf=$A/cmd_$(basename "$log" .log).sh; printf '%s\n' "$*" > "$cf"
  if [ "$MODE" = slurm ]; then sbatch --wait -p "$SBATCH_PARTITION" -A "$SBATCH_ACCOUNT" $GPU_OPTS $QOS_OPTS --cpus-per-task="$CPUS" --mem="$MEM" -t "${TIME_STEP:-01:00:00}" \
       -o "$log" --export=ALL,BUNDLE="$B",CMDFILE="$cf" "$B/jobs/acc_step.sbatch" || echo "sbatch rc $? for $log"; else MEMLOG=$RUN/logs/mem_$(basename "$log" .log).log bash "$cf" > "$log" 2>&1; fi; }
: > "$A/verdicts.txt"
# ---------------- 1 env
if step_on 1; then
  gpu_run "$A/01_setup.log" "cd $B && CUDA_FLAVOR=${CUDA_FLAVOR:-} OFFLINE=${OFFLINE:-1} WITH_LARGE=1 bash env/setup_env.sh && $PY acceptance/env_check.py"
  grep -h "ENVCHECK" "$A/01_setup.log" > "$A/01_envcheck.txt" || true
  if grep -q SETUP_OK "$A/01_setup.log" && grep -q '"verdict": "PASS"' "$A/01_envcheck.txt"; then pass 1 "$(grep -o '"gpus": \[[^]]*\]' "$A/01_envcheck.txt" | head -1) $(grep -o '"sdpa": {[^}]*}' "$A/01_envcheck.txt")"
  else fail 1 "see $A/01_setup.log ($(grep -o '"fail": \[[^]]*\]' "$A/01_envcheck.txt" | head -1))"; fi
fi
# ---------------- 2 parity16
if step_on 2; then
  gpu_run "$A/02_parity16.log" "cd $B && $PY $ENG parity16"
  grep -h "PARITY16" "$A/02_parity16.log" > "$A/02_parity16.txt" || true
  if grep -q "PARITY16 PASS" "$A/02_parity16.txt"; then pass 2 "$(cat "$A/02_parity16.txt")"; else fail 2 "$(tail -3 "$A/02_parity16.log")"; fi
fi
# ---------------- 3 bench grid (one process per config; each writes a BENCH line to $RUN/bench.jsonl)
if step_on 3; then
  rm -f "$RUN/bench.jsonl"; CM=imp; [ -f data/km_importance_412_janjul.tsv ] || fail 3 "data/km_importance_412_janjul.tsv missing: no silent fallback to random columns (shipped with the repository)"
  CMD="cd $B; for cfg in 'medium 20000 412' 'medium 20000 100' 'medium 50000 100' 'medium 50000 412' 'large 20000 100' 'large 20000 412' 'large 50000 100' 'large 50000 412'; do set -- \$cfg; echo \"=== bench \$1 \$2 \$3\"; NAME=BENCH SIZE=\$1 CAP=\$2 NCOL=\$3 COLMODE=$CM CORE=50 NINST=8 EIN=1 QMAX=${QMAX:-4096} $PY $ENG bench:janjul:LSZH || echo BENCH_FAILED \$1 \$2 \$3; done"
  [ "${LONGBENCH:-0}" = 1 ] && CMD="$CMD; echo '=== longbench whole pool LTFM 100 cols'; NAME=BENCH SIZE=medium CAP=1000000 NCOL=100 COLMODE=$CM CORE=50 NINST=8 EIN=1 QMAX=${QMAX:-4096} $PY $ENG bench:janjul:LTFM || echo BENCH_FAILED longbench"
  gpu_run "$A/03_bench.log" "$CMD"; grep -h "^BENCH " "$A/03_bench.log" | grep -v '"cap": 1000000' > "$A/03_bench.txt" || true; N=$(wc -l < "$A/03_bench.txt"); grep -h '"cap": 1000000' "$A/03_bench.log" > "$A/03_longbench.txt" || true
  if [ "$N" = 8 ] && ! grep -q "BENCH_FAILED\|MEMGUARD" "$A/03_bench.log"; then pass 3 "8/8 configs ($("$PY" -c "import json;L=[json.loads(l[6:]) for l in open('$A/03_bench.txt')];print(' | '.join(f\"{d['size'][0]}{d['cap']//1000}k/{d['ncol']} {d['ms_per_row']}ms {d['peak_gib']}GiB\" for d in L))"))"
  else fail 3 "$N/8 configs; $(grep -c BENCH_FAILED "$A/03_bench.log") failed, guard $(grep -c MEMGUARD "$A/03_bench.log")"; fi
fi
# ---------------- 4 scaling (4 units, one per GPU) + collision check + resume check
if step_on 4; then
  rm -rf "$RUN/locks"; T0=$(date +%s)
  if [ "$MODE" = slurm ]; then
    sbatch --wait -p "$SBATCH_PARTITION" -A "$SBATCH_ACCOUNT" $GPU_OPTS $QOS_OPTS --cpus-per-task="$CPUS" --mem="$MEM" -t "${TIME_STEP:-01:00:00}" --array=0-3 \
      -o "$A/04_scale_%a.log" --export=ALL,BUNDLE="$B",LIST="$RUN/lists/scale4.txt",PROFILE=shared20k,AMP=fp16,NAME=kumo_member_screen_km "$B/jobs/km_array.sbatch"
  else
    NG=$(nvidia-smi -L | wc -l); [ "$NG" -lt 4 ] && echo "only $NG GPU(s): scaling runs on $NG"
    for g in $(seq 0 $((NG - 1))); do CUDA_VISIBLE_DEVICES=$g MEMLOG=$RUN/logs/mem_scale_gpu$g.log $PY $TASK --from-list "$RUN/lists/scale4.txt" --claim > "$A/04_scale_$g.log" 2>&1 & done; wait
  fi
  WALL=$(( $(date +%s) - T0 )); NF=0; SUMS=0
  for u in $U4; do f=$RUN/results/member/kumo_member_screen_km_$(echo $u | cut -d: -f2)_$(echo $u | cut -d: -f3).npz; [ -f "$f" ] && NF=$((NF + 1)); done
  SUMS=$("$PY" - "$RUN" $U4 <<'PY'
import sys, json, numpy as np; R = sys.argv[1]; s = 0
for u in sys.argv[2:]:
    _, f, a = u.split(":"); p = f"{R}/results/member/kumo_member_screen_km_{f}_{a}.npz"
    try: s += json.loads(str(np.load(p)["meta"]))["secs"]
    except Exception: pass
print(int(s))
PY
)
  NDUP=$(cat "$A"/04_scale_*.log | grep -c "member\[kumo_member_screen_km\]" || true); SP=$("$PY" -c "print(round($SUMS/max(1,$WALL),2))")
  echo "scaling: $NF/4 files, $NDUP unit completions logged, sum of unit secs $SUMS, wall $WALL s, speed-up $SP" | tee "$A/04_scaling.txt"
  # resume: kill an engine instance after RESUME_KILL_S seconds, re-run, require rows_predicted_now < query (part files were reused)
  RK=${RESUME_KILL_S:-45}; RD=$RUN/results/engine/ACCRES; rm -rf "$RD"
  RCMD="cd $B && NAME=ACCRES SIZE=medium CAP=20000 NCOL=100 COLMODE=rand NINST=1 EIN=1 SHARD=2048 $PY $ENG inst:janjul:LSZH:0"
  if [ "$MODE" = slurm ]; then gpu_run "$A/04_resume_kill.log" "timeout -s KILL $RK bash -c '$RCMD'; true"; else timeout -s KILL $RK bash -c "$RCMD" > "$A/04_resume_kill.log" 2>&1; fi
  NP=$(ls "$RD"/janjul_LSZH/inst0.p*.npz 2>/dev/null | wc -l); echo "resume: killed after $RK s, $NP part files on disk" | tee "$A/04_resume.txt"
  gpu_run "$A/04_resume_rerun.log" "$RCMD"
  "$PY" - "$RD/janjul_LSZH/inst0.npz" "$NP" <<'PY' | tee -a "$A/04_resume.txt"
import sys, json, numpy as np; p, npart = sys.argv[1], int(sys.argv[2])
try:
    z = np.load(p); m = json.loads(str(z["meta"])); ok = np.isfinite(z["g"]).all() and len(z["g"]) == m["query"] and (m["rows_predicted_now"] < m["query"] or npart == 0)
    print(f"resume rerun: query {m['query']} rows_predicted_now {m['rows_predicted_now']} (parts reused: {m['query'] - m['rows_predicted_now']}) finite {bool(np.isfinite(z['g']).all())} -> {'RESUME_OK' if ok and npart > 0 else ('RESUME_OK_NO_PARTS (kill came before the first part; raise RESUME_KILL_S)' if ok else 'RESUME_FAIL')}")
except Exception as ex: print(f"RESUME_FAIL {ex}")
PY
  if [ "$NF" = 4 ] && [ "$NDUP" -le 4 ] && "$PY" -c "import sys; sys.exit(0 if $SP >= ${SCALE_MIN:-3.0} else 1)" && grep -q "RESUME_OK" "$A/04_resume.txt"; then pass 4 "$(cat "$A/04_scaling.txt") | $(grep RESUME "$A/04_resume.txt")"
  else fail 4 "$(cat "$A/04_scaling.txt") | $(grep -h RESUME "$A/04_resume.txt" | tail -1)"; fi
fi
# ---------------- 5 arm M stage 2 (remaining 16 units; resume skips the 4 of step 4) + precollect
if step_on 5; then
  T0=$(date +%s)
  if [ "$MODE" = slurm ]; then
    sbatch --wait -p "$SBATCH_PARTITION" -A "$SBATCH_ACCOUNT" $GPU_OPTS $QOS_OPTS --cpus-per-task="$CPUS" --mem="$MEM" -t "${TIME_STEP:-01:00:00}" --array=0-19%${MAX_PAR:-4} \
      -o "$A/05_armM_%a.log" --export=ALL,BUNDLE="$B",LIST="$RUN/lists/armM.txt",PROFILE=shared20k,AMP=fp16,NAME=kumo_member_screen_km "$B/jobs/km_array.sbatch"
  else
    rm -rf "$RUN/locks"; NG=$(nvidia-smi -L | wc -l)
    for g in $(seq 0 $((NG - 1))); do CUDA_VISIBLE_DEVICES=$g MEMLOG=$RUN/logs/mem_armM_gpu$g.log $PY $TASK --from-list "$RUN/lists/armM.txt" --claim > "$A/05_armM_$g.log" 2>&1 & done; wait
  fi
  echo "armM wall $(( $(date +%s) - T0 )) s" | tee "$A/05_armM_wall.txt"
  "$PY" acceptance/precollect.py "$RUN" kumo_member_screen_km months39,months410 | tee "$A/05_precollect.txt"
  if grep -q '"verdict": "PASS"' "$A/05_precollect.txt"; then pass 5 "arm M stage 2 fits complete: $(ls "$RUN"/results/oof/) ($(cat "$A/05_armM_wall.txt"))"; else fail 5 "$(grep -o '"fail": \[[^]]*\]' "$A/05_precollect.txt")"; fi
fi
summarize; echo "ACCEPTANCE_DONE $(date -u)"
