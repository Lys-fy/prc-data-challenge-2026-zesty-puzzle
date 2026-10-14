# SPDX-License-Identifier: GPL-3.0-only
# Common environment for the FM-engine jobs on the build machine. Sourced by the kumo_engine_j_*.sh / kumo_kcat_run.sh scripts (cd to the repo root first).
# Safe GPU settings: fp16 autocast, estimator_batch_size=1, SDM_CHUNK_MEMORY_FRACTION=0.02, expandable segments,
# one memory line per minute to MEMLOG, host-memory guard at MEMGUARD_GIB (exit 3, resumable). Payload = KUMO_CLUSTER_SETUP's staged frames.
# Root = $PRC_ROOT (else the current directory); Kumo env = $KUMO_PY (default .venv-kumo/bin/python, env/README.md).
R=${PRC_ROOT:-$PWD}
export PAYLOAD=${PAYLOAD:-$R/scratchpad/xclu/payload} RUN=${RUN:-$R/scratchpad/xkfm/run}
export HF_HOME=$PAYLOAD/vendor/hf HF_HUB_OFFLINE=1
export SDM_CHUNK_MEMORY_FRACTION=0.02 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MEMLOG=$R/scratchpad/xkfm/mem_${kumo_engine_job:-job}.log MEMGUARD_GIB=${MEMGUARD_GIB:-10} MEMLOG_SEC=60
export PRC_THREADS=${PRC_THREADS:-4} AMP=${AMP:-fp16}
KPY=${KUMO_PY:-$R/.venv-kumo/bin/python}; VPY=$R/.venv/bin/python; ENG=kumo_gpu/cluster/km_engine.py; TASK=kumo_gpu/cluster/km_task.py
mkdir -p "$RUN" "$R/scratchpad/xkfm"
APTS="LTFM EHAM EGLL LFPG EDDF LEMD LEBL EDDM LIRF LSZH"
echo "KUMO_ENGINE job ${kumo_engine_job:-?} host $(hostname) $(date -u) | AMP $AMP | MEMLOG $MEMLOG guard $MEMGUARD_GIB GiB | $(free -g 2>/dev/null | awk '/Mem:/{print "host mem total "$2" avail "$7" GB"}')"
