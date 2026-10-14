#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# KUMO_CLUSTER_SETUP: build the Kumo environment on a cluster login node or a GPU machine, fetch/verify the weights, self-check on the GPU.
# usage: bash env/setup_env.sh                 (from the bundle root; idempotent)
# env:   ENVDIR (default $PWD/.venv-kumo)   PAYLOAD (default $PWD/data/payload)   CUDA_FLAVOR (auto: cu126 for sm_70 / V100, else cu128)
#        WITH_SMALL=1 (also fetch small/regressor.pt for the optional fine-tune); large/regressor.pt is checked when PROFILE=l50* or WITH_LARGE=1   SKIP_VENV=1 ENV_PY=<python> (conda env given)
#        OFFLINE=1 (no GitHub/HF access: sdm from payload/vendor tarball, weights only from payload/vendor/hf)
# Network used (only when not OFFLINE): pypi.org, download.pytorch.org, github.com/NVIDIA/structured-data-models, huggingface.co/nvidia/Kumo-Tabular.
# Nothing of ours is uploaded anywhere by this script.
set -euo pipefail
B=$(cd "$(dirname "$0")/.." && pwd); cd "$B"
ENVDIR=${ENVDIR:-$B/.venv-kumo}; PAYLOAD=${PAYLOAD:-$B/data/payload}; export HF_HOME=${HF_HOME:-$PAYLOAD/vendor/hf}
SDM_COMMIT=ee6acd402176b0efa42527babd63f15672d63418; HF_COMMIT=bd7fa122b516c7355583873ffcf38f5e79403ebd
echo "== KUMO_CLUSTER_SETUP setup | host $(hostname) | $(uname -m) | $(date -u)"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader; CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1); else CC=?; echo "no nvidia-smi on this node (login node?): flavor must be given"; fi
FLAV=${CUDA_FLAVOR:-$([ "$CC" = "7.0" ] && echo cu126 || echo cu128)}; echo "torch flavor $FLAV (compute cap $CC)"
if [ "${SKIP_VENV:-0}" != 1 ]; then
  command -v uv >/dev/null || { echo "uv missing: 'pip install --user uv' or https://docs.astral.sh/uv/ (or use env/environment.yml)"; exit 2; }
  [ -x "$ENVDIR/bin/python" ] || uv venv --python 3.12 "$ENVDIR"
  PY=$ENVDIR/bin/python
  if [ "${OFFLINE:-0}" != 1 ]; then
    uv pip install --python "$PY" torch==2.11.0 --index-url https://download.pytorch.org/whl/$FLAV --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
      || { echo "torch 2.11.0/$FLAV not on the index: try CUDA_FLAVOR=cu128 or a torch>=2.7 version that has $FLAV wheels (sdm needs torch>=2.7)"; exit 3; }
    grep -v -E '^(#|torch==|triton==)' env/requirements.lock.txt | sed '/^$/d' > /tmp/xclu_req.$$; uv pip install --python "$PY" -r /tmp/xclu_req.$$; rm -f /tmp/xclu_req.$$
  fi
else PY=${ENV_PY:?set ENV_PY}; fi
# structured-data-models at the pinned commit (git, else the vendored pure-Python copy from the build machine)
if ! "$PY" -c "import sdm" 2>/dev/null; then
  if [ "${OFFLINE:-0}" != 1 ] && uv pip install --python "$PY" --no-deps "git+https://github.com/NVIDIA/structured-data-models.git@$SDM_COMMIT"; then echo "sdm from git @ ${SDM_COMMIT:0:7}";
  else SP=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); tar xzf "$PAYLOAD/vendor/sdm_ee6acd4_site.tgz" -C "$SP"; echo "sdm from vendored tarball -> $SP"; fi
fi
"$PY" - <<EOF
import json, glob, os; d = glob.glob(os.path.join(__import__("sysconfig").get_paths()["purelib"], "structured_data_models-*.dist-info/direct_url.json"))
c = json.load(open(d[0]))["vcs_info"]["commit_id"] if d else "?"; assert c == "$SDM_COMMIT", f"sdm commit {c} != pinned"; print("sdm commit OK", c[:9])
EOF
# weights: Kumo-Tabular v1.0.0 (= commit bd7fa122), sha256-checked
"$PY" - <<EOF
import os, hashlib
from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
want = {l.split()[1]: l.split()[0] for l in open("env/weights.sha256") if l.strip() and not l.startswith("#")}
files = ["medium/regressor.pt", "LICENSE"] + (["large/regressor.pt"] if os.environ.get("PROFILE", "m20e8").startswith("l") or os.environ.get("WITH_LARGE") == "1" else []) \
        + (["small/regressor.pt"] if os.environ.get("WITH_SMALL") == "1" else [])
for f in files:
    try: p = hf_hub_download("nvidia/Kumo-Tabular", f, revision="v1.0.0", local_files_only=True)
    except Exception:
        assert os.environ.get("OFFLINE") != "1", f"{f} not in {os.environ['HF_HOME']} and OFFLINE=1"
        p = hf_hub_download("nvidia/Kumo-Tabular", f, revision="v1.0.0")
    assert "/snapshots/$HF_COMMIT/" in p, f"v1.0.0 resolved to another commit: {p}"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if f in want: assert h == want[f], f"sha256 mismatch {f}: {h}"; print(f"weights OK {f} {h[:12]}")
    else:
        meta = get_hf_file_metadata(hf_hub_url("nvidia/Kumo-Tabular", f, revision="$HF_COMMIT")); assert meta.etag.strip('"') == h, (f, h, meta.etag)
        print(f"weights OK (Hub LFS sha256) {f} {h}  <- add to env/weights.sha256")
EOF
# GPU self-check (skipped on a login node without GPU)
if [ "$CC" != "?" ]; then HF_HUB_OFFLINE=1 "$PY" - <<'EOF'
import torch, sdm, numpy as np, time
cap = torch.cuda.get_device_capability(0); arch = f"sm_{cap[0]}{cap[1]}"; al = torch.cuda.get_arch_list()
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0), arch, "| wheel archs", al)
if arch not in al: print(f"WARNING: {arch} not compiled in this wheel (runs via a same-major cubin/PTX if present; B300 sm_103 on cu128 is expected to use sm_100)")
m = sdm.models.KumoTabular(task="regression", size="medium", device="cuda"); rng = np.random.default_rng(0)
x = rng.normal(size=(3000, 12)).astype(np.float32); y = (50 * x[:, 0] + 20 * x[:, 1] * x[:, 2] + rng.normal(0, 10, 3000)).astype(np.float32)
t = time.time(); m.fit(torch.as_tensor(x[:2000], device="cuda"), torch.as_tensor(y[:2000, None], device="cuda"), num_estimators=1, generator=torch.Generator(device="cuda").manual_seed(0))
p = m.predict(torch.as_tensor(x[2000:], device="cuda")); q = p.numerical if hasattr(p, "numerical") else p; g = q.float().mean(-1).cpu().numpy()
r = float(np.sqrt(((g - y[2000:]) ** 2).mean())); print(f"GPU smoke rmse {r:.2f} (noise sd 10; build machine CPU smoke 10.57) {time.time()-t:.1f}s | max mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
assert r < 16, "smoke rmse too high"
with torch.autocast("cuda", dtype=torch.float16):   # primary-profile path: fp16 + disjoint per-estimator contexts [E, cap, D], num_estimators=None
    xc = torch.as_tensor(x[:2000].reshape(4, 500, 12), device="cuda"); yc = torch.as_tensor(y[:2000].reshape(4, 500, 1), device="cuda")
    m.fit(xc, yc, num_estimators=None, estimator_batch_size=1, generator=torch.Generator(device="cuda").manual_seed(0)); xq = torch.as_tensor(x[2000:], device="cuda")
    try: p = m.predict(xq.unsqueeze(0).expand(4, *xq.shape)); how = "3-D query"
    except Exception as ex: p = m.predict(xq); how = f"2-D query (3-D: {type(ex).__name__})"
q = (p.numerical if hasattr(p, "numerical") else p).float()
while q.dim() > 2: q = q[0]
r2 = float(np.sqrt(((q.mean(-1).cpu().numpy() - y[2000:]) ** 2).mean())); print(f"GPU smoke fp16 per-estimator contexts ({how}): rmse {r2:.2f}")
assert r2 < 20, "fp16 per-estimator smoke rmse too high"; print("SETUP_OK")
EOF
fi
echo "env: $PY   HF_HOME=$HF_HOME"
