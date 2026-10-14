# Kumo Tabular GPU members (K250, KSTK)

Two members of the final model run NVIDIA's Kumo Tabular in-context learner on a GPU:
- **K250** (profile `KSBWPc100`): for each airport, the context is that airport's whole 2025 training pool (out-of-fold for the 2025 folds),
  split into one slice per estimator; the query rows are the airport's rows to predict. Output: the mean of the 999 output quantiles
  (`taxi`) and the spread between the estimators (used by KRS).
- **KSTK** (`kstk_profiles.txt`): Kumo reads the chain stacker's own meta frame (dumped by `scripts/kstk_run.sh dump`) and acts as a second
  stacking engine; its output `gK` is collected by `scripts/kstk_run.sh gk`.

The unmatched-row Kumo passes (F1, F3 on the 8,395 other unmatched rows, F4 on the 523 unmatched-LIRF rows of the final file) run on the CPU inside the main pipeline (`scripts/kumo_icl_unmatched.py`, `kumo_icl_lirf.py`).

## Environment and weights
`kumo_gpu/env/setup_env.sh` builds the environment (or `env/README.md`, kumo env): NVIDIA `structured-data-models` at commit
`ee6acd402176b0efa42527babd63f15672d63418`, weights `nvidia/Kumo-Tabular` tag `v1.0.0` (commit `bd7fa122…`), medium regressor
`medium/regressor.pt` sha256 `daa6a174…fb725` (`kumo_gpu/env/weights.sha256`). Weight licence OpenMDW-1.1. The weights are
downloaded by the user and run offline (`HF_HUB_OFFLINE=1`).

## Running
1. **Stage the inputs** on the build machine: `.venv/bin/python kumo_gpu/stage/stage_data.py` writes the per-airport, per-fold payload
   (feature matrices restricted to the importance-ordered columns of `kumo_gpu/data/km_features_412.txt`, 2025 labels for the
   context) into `kumo_gpu/data/payload/` (≈ 2 GB; competition-derived, never published).
2. **Run** on any CUDA machine:
   - plain GPU box (1–8 GPUs): `bash kumo_gpu/cluster/run_box.sh` inside tmux;
   - Slurm: fill the placeholders of `kumo_gpu/jobs/config.env` (partition, account, GPU type), then
     `bash kumo_gpu/jobs/submit_all.sh` (array jobs `km_array.sbatch`), or the staged form `MODE=slurm bash kumo_gpu/acceptance/run_acceptance.sh`.
   Units are (profile, fold or `prod`, airport); runs resume and lock per unit.
3. **Collect** on the build machine: `kumo_gpu/collect/collect.py` (member profiles) and `kumo_gpu/collect/reagg_local.py`
   (instance-level profiles such as `KSBWPc100`: the per-estimator outputs are re-aggregated into value and spread) write
   `data/oof/KSBWPc100_<fold>.npy`, `data/oof/KSBWPc100_prod.npy` and `data/preds/KSBWPc100P_s0.parquet`;
   `kumo_gpu/collect/check_member.py` validates them (finite, ids, row counts).

## Cost
K250: 6 folds + production ≈ 3–5 h per pass on one GPU workstation (the build machine), 30–45 min on one H100 for the production pass. KSTK `gK`:
1.5–2.5 h on the GPU workstation, 10–15 min on an H100. Final run (670,790 rows): K250 2.0 h, KSTK 2.3 h on the GPU workstation. Memory with fp16 inference,
measured on the public-period query set: ≤ 13 GiB GPU (LTFM, the largest pool, one instance) and < 20 GB host per process. Profile options are in
`kumo_gpu/acceptance/sb_profiles.txt`.
