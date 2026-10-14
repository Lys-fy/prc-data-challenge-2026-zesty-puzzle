# Licences of third-party components

The code in this repository is released under the **GNU GPL v3** (`LICENSE`). The repository does **not** redistribute any third-party
package, model weight or dataset: each is installed or downloaded by the user from its official source with the commands in `env/README.md`
and `docs/DATA_SOURCES.md`. The full per-package tables, read from the installed package metadata on the build machine on 2 Oct 2026, are in
`env/licences-main.tsv`, `env/licences-torch.tsv` and `env/licences-kumo.tsv`.

## 1. Software the model depends on
| component | version | licence | GPLv3-compatible use | role |
|---|---|---|---|---|
| LightGBM | 4.7.0 | MIT | yes | members, stacker, stages |
| CatBoost | 1.2.10 | Apache-2.0 | yes | members |
| polars | 1.44.1 / 1.44.2 | MIT | yes | all data handling |
| numpy | 2.5.2 / 2.5.3 | BSD-3-Clause (+ 0BSD, MIT, Zlib, CC0 parts) | yes | |
| pandas | 3.0.5 | BSD-3-Clause | yes | |
| pyarrow | 25.0.1 | Apache-2.0 | yes | parquet I/O |
| scikit-learn | 1.9.0 / 1.9.1 | BSD-3-Clause | yes | |
| scipy | 1.18.1 | BSD-3-Clause | yes | |
| minio | 7.2.20 | Apache-2.0 | yes | competition-data download |
| requests / tqdm | 2.34.2 / 4.70 | Apache-2.0 / MPL-2.0 AND MIT | yes | downloads |
| PyTorch | 2.11.0+cu128 | BSD-3-Clause | yes | RealMLP members, BIG set transformer, Kumo |
| pytabkit | 1.7.3 | Apache-2.0 | yes | RealMLP members |
| NVIDIA structured-data-models (`sdm`) | git `ee6acd402176b0efa42527babd63f15672d63418` | Apache-2.0 | yes | Kumo Tabular runtime |
| huggingface_hub / safetensors | 1.32.0 / 0.8.0 | Apache-2.0 | yes | weight download |
| triton | 3.6.0 | MIT | yes | PyTorch dependency |
| NVIDIA CUDA runtime wheels (cublas, cudnn, nccl, …) | 12.8 | NVIDIA proprietary (redistributable runtime EULA) | not redistributed; installed by pip with PyTorch | GPU runtime |
| uv | 0.11.14 | MIT OR Apache-2.0 | tool only | environment manager |

Also present in the main environment but not on the model's path: duckdb (MIT), shapely / pyproj (BSD-3 / MIT; only the unused `geo` group),
networkx (BSD-3), xgboost (Apache-2.0), matplotlib (PSF-based; figures only). The build machine's freeze `env/requirements-main.txt` also
lists optuna, seaborn, jupyterlab and pyopensky, which no script imports and `pyproject.toml` does not declare.

**Check result.** Every package in the three environments carries an OSI-approved permissive or weak-copyleft licence (MIT, BSD, Apache-2.0,
ISC, PSF, MPL-2.0), except the NVIDIA CUDA runtime wheels, which are proprietary but freely redistributable runtime libraries; none of them is
copied into this repository. Apache-2.0 and MPL-2.0 components are compatible with use from GPLv3 code.

## 2. Pre-trained model weights
| weights | source | revision | file used | sha256 | licence |
|---|---|---|---|---|---|
| Kumo Tabular (medium regressor) | Hugging Face `nvidia/Kumo-Tabular` | tag `v1.0.0` = commit `bd7fa122b516c7355583873ffcf38f5e79403ebd` | `medium/regressor.pt` | `daa6a17420b5aed231007e5d2f9c712abe093b4f94b3e1189b178947a95fb725` | **OpenMDW-1.1** (permissive: use, modify, distribute; keep the licence and notices when redistributing) |

The weights are downloaded by the user and not redistributed. They run locally (no API call), only for in-context prediction: on the
unmatched rows (`scripts/kumo_icl_unmatched.py`, `scripts/kumo_icl_lirf.py`, CPU) and as the K250 and KSTK members (`kumo_gpu/`, GPU).

## 3. Data
Dataset licences, attribution and the ODbL share-alike note: `docs/DATA_SOURCES.md`.
