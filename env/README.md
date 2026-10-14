# Environments

All three environments were built with **uv 0.11.14** from the system **Python 3.12.3** on the build machine:
aarch64 (20 ARM cores), NVIDIA GB10 GPU (driver 580.126.09, CUDA 12.8), unified memory (`docs/REPRODUCE.md` §1).
17 seeds of 6 LightGBM member families were trained on a second host (x86_64, CPU only): queue_decomp_ebqq s3–5, lgb_adsb_linked_a
s0–2, lgb_noclock_nc s3–5, lgb_airport_ltfm s3–5, lgb_airport_lfpg s3–5 and pseudolabel_pl3p88 s0–1 (`arch` column of
`phase2/manifest/members.tsv`).

System dependencies: `libgomp1` (OpenMP runtime of LightGBM; `apt install libgomp1` on Debian/Ubuntu), `bash`, `make`, `git`; an
NVIDIA driver with CUDA 12.8 for the GPU members.

| env | used for | pin | how to create |
|---|---|---|---|
| main (`.venv`) | features, every LightGBM/CatBoost member, all stacking and stage scripts, the combine tool | `uv.lock` (repo root) + `env/requirements-main.txt` (freeze of the build machine). Key pins: lightgbm 4.7.0, catboost 1.2.10, polars 1.44.1, numpy 2.5.2, scikit-learn 1.9.0 | `uv sync` |
| torch (`$TORCH_PY`, `.venv-torch`) | the RealMLP neural members (NNRA2, meta_oof_realmlp_nnra3) and the BIG airport-day set transformer (`scripts/dayset_big_member.py`) | `env/requirements-torch.txt`. torch 2.11.0+cu128, pytabkit 1.7.3 | `uv venv --python python3.12 .venv-torch && uv pip install --python $TORCH_PY -r env/requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match` |
| kumo (`$KUMO_PY`, `.venv-kumo`) | the Kumo Tabular in-context learner: on the unmatched rows (F1, F3) and the unmatched LIRF rows (F4) on the CPU; the K250, KCAT8 / KCATL and KSTK members on a GPU (`docs/kumo_gpu.md`, same pins; `kumo_gpu/env/setup_env.sh` builds it on a GPU box) | `env/requirements-kumo.txt`; see below | see below |

## Kumo environment (exact pins)
- **Library:** NVIDIA `structured-data-models` at git commit **`ee6acd402176b0efa42527babd63f15672d63418`** (installed version
  `0.1.0rc2.dev177+gee6acd402`, licence Apache-2.0).
- **Weights:** Hugging Face `nvidia/Kumo-Tabular`, tag **`v1.0.0`**, which resolves to snapshot commit
  **`bd7fa122b516c7355583873ffcf38f5e79403ebd`**. The library pins `revision="v1.0.0"` itself (`sdm/models/kumo/tabular/model.py`).
  - Only the **medium regressor** is used: `medium/regressor.pt`, sha256
    **`daa6a17420b5aed231007e5d2f9c712abe093b4f94b3e1189b178947a95fb725`**.
  - Weight licence: **OpenMDW-1.1** (the `LICENSE` file of the snapshot, sha256 `44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d`).
- **Other pins:** torch 2.11.0+cu128, numpy 2.5.3, polars 1.44.2, pyarrow 25.0.1, huggingface_hub 1.32.0, safetensors 0.8.0.
- **Create:**
  ```bash
  uv venv --python python3.12 .venv-kumo
  uv pip install --python $KUMO_PY torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
      --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
  uv pip install --python $KUMO_PY polars==1.44.2 numpy==2.5.3 pyarrow==25.0.1 huggingface_hub==1.32.0 \
      "git+https://github.com/NVIDIA/structured-data-models.git@ee6acd402176b0efa42527babd63f15672d63418"
  # weights, once (then the jobs run offline):
  HF_HOME=$PRC_ROOT/artifacts/xkumo/hf $KUMO_PY -c \
      "from huggingface_hub import snapshot_download as s; print(s('nvidia/Kumo-Tabular', revision='v1.0.0'))"
  sha256sum $PRC_ROOT/artifacts/xkumo/hf/hub/models--nvidia--Kumo-Tabular/snapshots/bd7fa122*/medium/regressor.pt   # must be daa6a174...fb725
  ```
- **Run settings** (step `kumo_env` of `phase2/run_pipeline.sh`): `HF_HOME=$PRC_ROOT/artifacts/xkumo/hf HF_HUB_OFFLINE=1`,
  `KumoTabular(task="regression", size="medium")`, 4 estimators, seed 0. The prediction is the mean of the 999 output quantiles.
- **Device:** the F1, F3 and F4 files in the final submission were computed on the **CPU** (4 threads, aarch64); the library gives the same
  model on the GPU, with measured CPU/GPU differences around 1e-3 s. The K250, KCAT8 / KCATL and KSTK members ran on GPUs (`docs/kumo_gpu.md`).

## Not used by the final submission
AutoGluon Mitra and the TabICL checkpoint inside `structured-data-models` were tested but feed no row of the
final file, so they are not pinned here.
