# Reproducing the final submission

From the raw competition files to `zesty-puzzle_final.parquet` (670,790 departures, Jan + Feb + Jun + Jul 2026). Two parts:
- **A. 2025 side, run once:** 6-fold month OOF of every member, the fold side of every chain and chain variant, and the 2025 constants. Uses
  2025 labels only.
- **B. Production run on a ranking file:** features, side tables, members (fitted from scratch on all 2025 rows), Kumo members, chains,
  levers and rules for the ranking rows. Every stage refits its stage model on the A-side fold vectors.

## 1. Hardware
| machine | spec | used for |
|---|---|---|
| ASUS Ascent GX10 (NVIDIA DGX Spark platform) | NVIDIA GB10 Grace Blackwell superchip, 20-core Arm CPU (10× Cortex-X925 + 10× Cortex-A725), 128 GB unified LPDDR5x memory, of which about 119 GB is visible to the OS; CUDA 12.8 | Kumo GPU members, CatBoost GPU and neural members, the final pipeline; everything unless stated |
| Apple M5 Max | 18 cores, 64 GB | part of the LightGBM members (bit-identical to the GX10), one chain stacker, checks |
| AMD Ryzen 5 8500GE | x86_64, 6 cores / 12 threads | validation scoring, replays, 17 LightGBM seeds of 6 member families (`arch` column of `phase2/manifest/members.tsv`) |
| NVIDIA H100 (cloud) | one GPU | additional Kumo fold runs |
| 8× NVIDIA RTX PRO 6000 (cloud) | 96 GB each | production fits of 28 of the 42 CatBoost seeds of `scripts/xsave/members_xsave.tsv` (CATV17x s0–19, runway_queue_catv17xq s0–5, CATV9x s0–1; CATV9x s2–15, the seed bag of line B, on the GX10), RealMLP (8) and BIG (3): 2.9 h wall |

## 2. Runtimes (measured)
| step | time |
|---|---|
| adsb.lol download | 2–4 GB and 3–4 min per day, ≈ 485 days |
| clock extraction | ≈ 13 s per day, 6 days in parallel |
| features + side tables of a ranking file | ≈ 1.5 h |
| members, production fit of every seed on all 2025 rows | ≈ 330 CPU-h + ≈ 20 GPU-h: per seed on an RTX PRO 6000 CatBoost 18 min, RealMLP 53 min, BIG 5 min; 42 × 18 + 8 × 53 + 3 × 5 = 1,195 min |
| Kumo K250 on 670,790 rows | 2.0 h on the workstation GPU (30–45 min on one H100) |
| Kumo KCAT8 + KCATL on 670,790 rows (80 + 80 instances) | 55–129 s per instance, ≈ 3–4 h |
| base line (`FORM=v99`) incl. the pseudo-label second pass | ≈ 1.5 h |
| KSTK `gK` (2 chains) | 2.3 h on the workstation GPU |
| chain-variant levers, TW1′, KRS, KRS-NL, rules | ≈ 2 h (one chain variant ≈ 15–25 min at 6 threads) |
| bag line (base line with the 24 + 16 seed bag) | ≈ 2.2 h |
| final assembly | ≈ 10 s |
| A side | about one full member fit plus 1–2 h of fold fitting per chain |

## 3. Setup
```bash
export PRC_ROOT=$PWD PYTHONHASHSEED=0          # every script reads the root from PRC_ROOT (unset: the current directory)
make env                                       # main env .venv from uv.lock (Python 3.12); torch and Kumo envs: env/README.md
export TORCH_PY=$PWD/.venv-torch/bin/python KUMO_PY=$PWD/.venv-kumo/bin/python
```
| env | key pins |
|---|---|
| main `.venv` | Python 3.12.3, lightgbm 4.7.0, catboost 1.2.10, polars 1.44.1, numpy 2.5.2, scikit-learn 1.9.0 |
| torch `.venv-torch` (`$TORCH_PY`) | torch 2.11.0+cu128, pytabkit 1.7.3 |
| Kumo `.venv-kumo` (`$KUMO_PY`) | NVIDIA `structured-data-models` @ `ee6acd40…`, weights `nvidia/Kumo-Tabular` v1.0.0, `medium/regressor.pt` sha256 `daa6a174…fb725` (OpenMDW-1.1) |

## 4. Data
```bash
export PRC_ACCESS_KEY=<your-access-key> PRC_SECRET_KEY=<your-secret-key>    # team credentials from the organisers; never commit
make data            # -> data/raw/ (training_*.parquet, ranking.parquet); data/final/ (final_ranking.parquet, final_submitting.parquet)
make external        # METAR (IEM), OPDI v0.0.2 (~15 GB), PRU daily csv, OpenFlights   (docs/DATA_SOURCES.md)
make adsb-extract    # adsb.lol 2025 + Jan and Jul 2026 -> airport boxes -> per-departure clock day files (resumable)
```
`make data` writes the `final_*` objects to `data/final/` and the others to `data/raw/`; the two files of §8 are checked by md5 (an existing
file is skipped only when its md5 matches; a downloaded file that does not match stops the script). `make adsb-extract`
(`scripts/adsb_all.sh`) pulls the 2025 days and January and July 2026; the February and June 2026 days are pulled and extracted by the
`inputs` stage of the production run (`p2_final_inputs.sh final adsb_extract`, every day of the ranking file whose day file is missing).
`manifest/external_inputs.md5` lists size and md5 of every external file the final build read (PRU csv of `data/external/pru_20261008`,
the OPDI files of `OPDI_PERIODS`, the two METAR tables, the adsb.lol day files); a fresh download can differ. One PRU directory serves
both readers: `pru_build.py` gives the same `xpr_pru_daily.parquet` (`d1212ccd…`) from the 24 Sep, 25 Sep or 8 Oct download (day grid
ends 2026-07-31), and `pru_daily_calib.py` read `pru_20261008` in the final workspaces.

## 5. A. 2025 side (once)
Build A with the public ranking file in `data/raw/` (a few pooled statistics use training + ranking rows). Folds:
`months39 months28 months410 janjul months612 months511` (the 2025 month pairs 3+9, 2+8, 4+10, 1+7, 6+12, 5+11; `janjul` is the
January + July 2025 pair). `<C>` is a chain (dTK, Kq, NNA3).

`make features` builds `data/features_v9.parquet` and the side tables. `make folds` (`scripts/a_side.sh`, resumable, markers in
`scratchpad/a_side/`, `DRY=1` lists the commands) runs A1 and A2 below with these names; the KSTK fold Kumo runs go to
`kumo_gpu/run_kstk_folds` (`KSTK_FOLD_RUN`, read by `scripts/kumo_prod.sh kstkgk`).

**A1. Member OOF** (`data/oof/<name>_<fold>.npy`):
| member | command |
|---|---|
| every family of `phase2/manifest/members.tsv` | its `command` column with `--cv months:a,b` in place of `--predict-rank`, per fold |
| derived queue_decomp_ebdt, sixseed_chain_* | `scripts/queue_mkdt.py` |
| chain meta inputs queue_eb_ra3qa, meta_oof_h_ctl, NNA2, queue_decomp_ebqq, runway_queue_catq, meta_oof_realmlp_nna3 | `bash scripts/meta_oof_folds.sh <fold>` per fold; then `.venv/bin/python scripts/queue_mkdt_oof.py months39 months28 months410 janjul months511 months612` (queue_eb_dt_dt) |
| wideleaf_a2 | `bash scripts/wideleaf_member.sh A2s{0,1,2} <fold>` |
| pseudolabel_pl3 | `bash scripts/pseudo_member.sh` (fold form; teacher = the 2025 OOF of the chain line) |
| NBR_CLOCK MA / MBm | `.venv/bin/python scripts/nbrclock_member.py MA,MBm months39,months28,months410,janjul,months511,months612` |
| BIG | `$TORCH_PY scripts/dayset_big_member.py --fold <fold> --tag BIG[_s1,_s2] --d 192 --layers 4 --heads 6 --epochs 16 --seed {0,1,2}`, then `ARM=BIG .venv/bin/python scripts/dayset_prep.py` |
| K250 | `kumo_gpu/stage/stage_data.py`, then the Kumo engine with profile `KSBWPc100` (`kumo_gpu/acceptance/sb_profiles.txt`), 6 folds × 10 airports, then `kumo_gpu/collect/collect.py` (`docs/kumo_gpu.md`) |
| KCAT8 | codes `.venv/bin/python scripts/kumo_kcat_codes.py data/features_v9.parquet <fold payload>/member scratchpad/kcat/codes/folds`; `bash scripts/kumo_kcat_run.sh <fold>` per fold; `FOLDS="months39 months28 months410 months511 months612 janjul" bash scripts/kumo_kcat_collect.sh` |
| KCATL, KCATL16, KCATLg | `bash scripts/kumo_kcat_logtarget_run.sh folds`; `bash scripts/kumo_kcat_logtarget_bag.sh folds`; `.venv/bin/python scripts/kumo_kcat_logtarget_tailguard.py months39 months28 months410 months511 months612 janjul` |
| KSTK `gK` | `bash scripts/kstk_run.sh dump <C>` → Kumo with `kumo_gpu/acceptance/kstk_profiles.txt` → `bash scripts/kstk_run.sh gk <run> <C>` (needs the k250_stack chains of A2) |

**A2. Chain fold sides** (1–2 h per chain at 6–10 threads):
| chain | command |
|---|---|
| dTK / Kq / NNA3 | `bash scripts/chain_jobs/stack/j_T2x_<C>.sh`; clock stages `bash scripts/chain_jobs/clockstage/j_L.sh`, `j_Kq.sh`, `j_NNA3.sh` |
| XB, XS (Kq, NNA3) | `bash scripts/wideleaf_fold.sh <C>`; `bash scripts/pseudo_fold.sh PL3 <C>` |
| nbr_clock_twin_bm, NBR_CLOCK_TWIN | `KM=nbr_clock_mbm PFX=nbr_clock_twin_bm bash scripts/chainvar_fold.sh <C>`; the same with `KM=nbr_clock_ma PFX=NBR_CLOCK_TWIN` |
| k250_stack | `KM=KSBWPc100 PFX=k250_stack bash scripts/chainvar_fold.sh <C>` |
| KSTK_CLOCK | `KM=KSBWPc100 KMP=KSBWPc100P PFX=k250_stack bash scripts/kumo_clock_chain.sh <C>` |
| BIG_NOVELTY | `PFX=BIG_NOVELTY big=big_big3 BIG_PROD_PREFIX=big_bigp extra3_mask_ids=<novelty ids> bash scripts/dayset_chain.sh fold <C>` |
| kstk_input | `bash scripts/kstk_run.sh fold <C>` |
| kcatlg_stack | `KM=KCATLg PFX=kcatlg_stack bash scripts/chainvar_fold.sh <C>` |

**A3. 2025 constants.** Shipped as values; the fitting step is listed, with any input this repository does not rebuild. `make constants`
(`scripts/constants_2025.sh`, resumable, one marker per step) runs the steps below in order; `make meta-oof` runs the meta-input OOF of A1.
`REFIT_CONSTANTS=1 bash scripts/run_final.sh <period> inputs` refits TW1′, KRS, KRS-NL and the 2025 context of F4 (the Kumo in-context step on
unmatched LIRF rows) before the production run; without it the runner checks that the constants are present. The run reads the shipped
values (`phase2/params/`, TW1S = 1264 in `phase2/steps/p2_v107.sh`); the KRS-NL model and the F4 context are built from competition data
by `make constants` and are not published.
| constant | value / file | fitted by | inputs not rebuilt here |
|---|---|---|---|
| sealed days | `data/sealed_days.parquet` | `scripts/sealed_days.py` (fixed seed) | – |
| BIG novelty threshold | 2025 p99 per airport | `scripts/dayset_novelty.py` (label-free) | – |
| pool weights | .3212 / .3065 / .3722 (`phase2/params/pool_weights.json`) | `scripts/constants_2025.sh pool` → `scripts/pool_weights_fit.py`: global simplex over the 6-fold HA0 vectors F6, P12 dTK, P12 Kq, P12 RdLN, P12 NNA3 (F6 and RdLN get weight 0); reproduces the 4-decimal values exactly, the unrounded ones to ≈ 1e-5. The 5-slot `pool_mix` call uses the fitted NNA3 weight .3722; the 3-weight lists of `pru_daily_calib.py` use .3723 so that the weights sum to 1 | the five P12-generation fold vectors (`scratchpad/xof/`) |
| post-rule arms standfloor, offblock_pull | `phase2/params/postrules_fixed.json` | `scripts/postrule_fold_frame.py --pool scratchpad/xpr/pru_pxxb --out scratchpad/x4/X4_fold.parquet` → `phase2/steps/p2_postrules.py fit --fold scratchpad/xpr/pru_pxxb --x4 scratchpad/x4/X4_fold.parquet --out <json>`; shifts rounded to 1 s (reproduces every arm; offblock_pull shifts −47.33 / +34.62 → −47 / +35; the rebuilt X4 frame is not byte-identical to the saved one) | – |
| TW1′ shift | +1,264 s (`TW1S` in `phase2/steps/p2_v107.sh`) | `scripts/tw1_twins.py` → `bash scripts/error_atlas_line.sh` (the three 2025 input lines and `scripts/error_atlas_tables.py`) → `scripts/tw1_rule_build.py` (writes `scratchpad/xquick/tw1p_shift.json` for comparison) | – |
| KRS β | .4520 / .1726 / .0674 (`phase2/params/krs_coef.json`) | `bash scripts/krs_fold_line.sh` (fold line of the chain-variant part from the k250_stack / KSTK_CLOCK / BIG_NOVELTY / kstk_input chain fold sides of A2 at the (B) rung → `scripts/chain_post.py` → `scripts/krs_fit2.py` MODES=KRS; reproduces β and edges exactly, fold lines max \|diff\| 4e-12 s) | – |
| KRS-NL coefficients and model | `phase2/params/krsnl_coef.json`; model `scratchpad/xkrsnl/prod/gbm_A.txt` (not published) | `scripts/months_rows_table.py` (per-row fold table `scratchpad/xmonths/rows.parquet` from the krs_fold_line outputs) → `scripts/krs_nonlinear.py fit A` (month-kernel β max \|diff\| 2e-16; GBM predictions on the 2.06 M fold rows max \|diff\| 6e-14 s) | – |
| exposed-row bounds | 0.072 / 0.259 | `scripts/exposed_ord_bounds.py` (95 % exact upper bound of the ORD share, 0/40 crossers and 0/10 non-crossers in 2025) | – |
| E1s clock cap (own at-stand clock rule for exposed non-LIRF rows) | 1,822 s | `scripts/e1s_clock_cap.py` (largest at-stand clock of the 21 linked 2025 exposed non-LIRF rows) | – |
| F4 context | `scratchpad/xhr/unm2025b.parquet`, `unm_lirf_janjul_frozen.parquet` | `scripts/exposed_characterise.py` (2025 part) → `scripts/exposed_context.py`; `scripts/exposed_mechanisms.py` part 2 | – |

## 6. B. Production run on the final ranking file
The submitted file combines two runs of the line on the final ranking file: **line A**, the KCATLg line (`phase2/periods/final.env`,
default seed bag), gives the matched and other unmatched rows; **line B**, the bag line (`phase2/periods/final_bag.env`, the same line
with 24 LightGBM + 16 CatBoost seeds for the rule-priced rows), gives the unmatched-LIRF and exposed rows; the schedule-copy rule sets
69 LIRF rows. Two line runs exist because the rule-priced rows take the larger seed bag of line B while every other row takes line A;
the final assembly needs the two line outputs, line A's `final.v111.parquet` and line B's `final.v99.parquet` (with line B's row classes
and line A's features table, the four files of `manifest/final_inputs.md5`). `scripts/run_final.sh final [stage ...]` (`make final STAGE=<stage>`) runs one stage at a time; each stage resumes from its
step markers in `phase2/work/<ws>/logs/`, and the workspace directory is copied between machines when the hardware changes.

| stage | `run_final.sh` stage → scripts | hardware | runtime | inputs | outputs |
|---|---|---|---|---|---|
| inputs | `inputs` → `run_pipeline.sh final ws`, `p2_final_inputs.sh final opdi`, `run_pipeline.sh final np_features`, `p2_final_inputs.sh final side1 adsb_extract adsb_tables check`, `run_pipeline.sh final np_side`, `p2_v107.sh final v107link` | CPU | ≈ 1.5 h | competition files, external data (§4), A side (§5), `phase2/params/` | `phase2/work/final/`: `data/features_v9.parquet` (`c9a079ef…`), side tables, linked fold sides |
| members | `members` → `scripts/xsave/final_refit.sh` per seed of `scripts/xsave/members_xsave.tsv`, NBR_CLOCK MBm / MA, BIG s0–2, `scripts/members_check.sh` | CPU (LightGBM) + CUDA GPU (CatBoost, RealMLP, BIG) | ≈ 330 CPU-h + ≈ 20 GPU-h (§2) | features, side tables | `data/preds/<member>_s<k>.parquet`, `artifacts/xsave/` |
| kumo | `kumo` → `scripts/kumo_prod.sh final k250stage k250 k250collect kcat` | CUDA GPU | K250 2.0 h, KCAT8 + KCATL 3–4 h | features, Kumo weights | `data/preds/KSBWPc100P_s0.parquet`, `KCATLgP_s0.parquet` |
| line | `line` → `phase2/run_pipeline.sh final` (`FORM=v99`) | CPU | ≈ 1.5 h | members, Kumo members | `submissions/final.v99.parquet` (`f5570dc8…`) |
| kstk | `kstk` → `scripts/kumo_prod.sh final kstkdump kstkbundle kstk kstkgk` | CUDA GPU | 2.3 h | chain-stacker frames | `data/preds/KSTK6{Kq,NNA3}P.npy` |
| levers | `levers` → `phase2/steps/p2_v107.sh final` | CPU | ≈ 2 h | `final.v99`, K250, KCATLg, BIG, KSTK, NBR_CLOCK, `phase2/params/krs*_coef.json` | `submissions/final.v111.parquet` (line A, `e4354437…`) |
| bag | `bag` → copy of `phase2/work/final` after the members to `phase2/work/final_bag`, `phase2/run_pipeline.sh final_bag` | CPU | ≈ 2.2 h | line A's workspace after `kumo` | `phase2/work/final_bag/submissions/final.v99.parquet` (line B, `bc0fa068…`), `scratchpad/p2/U.parquet` (`137042cf…`) |
| assemble | `assemble` → `scripts/assemble_final.sh` (`make assemble STAGES=<dir>`) | CPU, 4 threads | ≈ 10 s | the four files of `manifest/final_inputs.md5` | `zesty-puzzle_final.parquet` (`1a155e9f…`), `public_janjul.parquet` (`9cd6296d…`) |

The final assembly from the two line outputs (stage dir with `kcatlg_line.parquet`, `bag_line.parquet`, `bag_classes.parquet`,
`features_final.parquet`; root with `data/raw/training_*.parquet`, `data/final/final_ranking.parquet` and, for F4, the public-period
`data/features_v9.parquet` `216667fb…`):
```bash
STAGES=<stage dir> bash scripts/assemble_final.sh <out dir>      # make assemble STAGES=<stage dir>; CHECK=0 skips the md5 checks
```
| step | script | output | md5 |
|---|---|---|---|
| F1 | `rule_schedule_copy_lirf_relink.py` (orphan relink from the raw files; 69 rows: Jan 23, Feb 5, Jun 17, Jul 24) | `schedule_copy/final_dryrun.parquet` | `ab87ba639475942eeb38ba042c90ae07` |
| F2 | `rulebag_rows_merge.py`: line A, with the 592 unmatched-LIRF and 70 exposed rows of line B | `F2_merge.parquet` | `34c4a6e58a4b87fce7f8081d20c7e965` |
| F3 | `rule_schedule_copy_lirf_apply.py`: the 69 schedule-copy rows; `validate_submission.py` | **`zesty-puzzle_final.parquet`** (670,790 rows) | **`1a155e9fc6e43cff09ca3b9f46b7cede`** |
| F4 | `public_period_extract.py`: Jan + Jul rows on the public ids, public-file LOBT bound (3 rows moved by ≤ 5 s) | `public_janjul.parquet` (344,841 rows; public leaderboard **221.4616**) | `9cd6296dabbf5190e2e79765a897c33b` |

## 6b. The staged assembly of the build
**Two runs and four runs.** The submitted bytes `1a155e9f…` are reproduced by the two-line assembly `scripts/assemble_final.sh` (§6):
line A = the KCATLg line, line B = the bag line. The original build was staged: it combined four saved runs, line A with K250, with KCAT8
and with KCATLg in the Kumo-member slot of the levers (`V102ARM`), plus line B. Its delta steps add (KCAT8 − K250) and then
(KCATLg − KCAT8), so the K250 and KCAT8 runs cancel and the result equals the two-line assembly byte for byte.
`scripts/assemble_final_staged.sh` replays the staged build (S1–S6 below). The git tag `submission-build` marks the code state of the build.
```bash
STAGES=<stage dir> bash scripts/assemble_final_staged.sh <out dir>       # ~1 min, 4 threads; CHECK=0 skips the md5 checks
```
| stage-dir file | what it is | md5 |
|---|---|---|
| `base_line.parquet` | `submissions/final.v111.parquet` of line A with K250 in the Kumo-member slot (`V102ARM=k250_stack`) | `373674bd6cde06f477891412b88369bf` |
| `bag_line.parquet` | line B, `submissions/final.v99.parquet` | `bc0fa06831e8660bf3a3d09bc165c100` |
| `bag_classes.parquet` | row classes of line B (`scratchpad/p2/U.parquet`: matched / unm_LIRF / exposed / unm_other) | `137042cf2700eb984888535a5d68b7c6` |
| `kcat8_line.parquet` | line A with KCAT8 in the Kumo-member slot | `2e23a351ec49123b131a11e4915e07d2` |
| `kcatlg_line.parquet` | line A (`V102ARM=kcatlg_stack`) | `e435443735aba545bef0499eabacf7c8` |
| `features_final.parquet` | `data/features_v9.parquet` of line A (LOBT bound) | `c9a079efaee3ae064582cd06307ad199` |

| step | script | output | md5 |
|---|---|---|---|
| S1 | `rule_schedule_copy_lirf_relink.py` | `schedule_copy/final_dryrun.parquet` | `ab87ba639475942eeb38ba042c90ae07` |
| S2 | `rulebag_rows_merge.py`: unm_LIRF and exposed rows from line B | `S2_rulebag.parquet` | `a0245491dd1262daddf98abfca1423a2` |
| S3 | `staged_delta_apply.py`: + (KCAT8 line − K250 line), 519,653 rows, 0 LOBT clips | `S3_kcat8.parquet` | `f7784e3e62a2b2d9199eee65ad0fd6a4` |
| S4 | `rule_schedule_copy_lirf_apply.py`: the 69 schedule-copy rows | `S4_schedule_copy.parquet` | `d95ad32d5a629b3ff7015924d3bcefa4` |
| S5 | `staged_delta_apply.py --changed-only --protect`: + (KCATLg line − KCAT8 line), 472,468 rows, delta 0 on all 9,057 rule-priced and schedule-copy rows | **`zesty-puzzle_final.parquet`** | **`1a155e9fc6e43cff09ca3b9f46b7cede`** |
| S6 | `public_period_extract.py` | `public_janjul.parquet` | `9cd6296dabbf5190e2e79765a897c33b` |

**Deviations of the submitted build** from a single run of this repository's code (`docs/known_issues.md` §6–§7):
| what happened | effect | command / file |
|---|---|---|
| `lgb_adsb_linked_a` s0–s2 are saved models fitted on the feature tables of the public period (pooled `arrfb`, target-encoding and queue statistics include the public ranking file) | the submitted file uses their bit-exact re-predictions; a refit on the final-period tables moves the LTFM level by −17 to −24 s on 89,793 rows | `scripts/xsave/final_refit.sh` (re-predict from `artifacts/xsave/`), `scripts/train2_linked.py`; `known_issues.md` §6 |
| 17 LightGBM seeds of 6 families, among them the pseudo-label member `pseudolabel_pl3p88` s0–1, were fitted on an x86_64 host; everything else on aarch64 | aarch64 and x86_64 build different trees; the saved outputs are inputs of §6c, a refit on another architecture is a new draw | `arch` column of `phase2/manifest/members.tsv`; `env/README.md` |
| line B ran its lever scripts from a separate source directory (`P2_LEVER_SRC`) | its replay from the iso stage on is byte-exact with this repository's scripts; the steps before it were not rerun | `phase2/steps/p2_v99_prep.py:26` |
| line B reran its base line from `np_derive`, including the pseudo-label member refit | its `final.v99` differs from line A's on 353,021 matched rows, so one workspace with both settings gives a different file | `phase2/periods/final_bag.env`, `scripts/run_final.sh final bag` |
| pool weights come from saved P12-generation fold vectors that this repository does not rebuild | the shipped .3212 / .3065 / .3722 are used as values; a fit on this repository's A2 chains gives .3338 / .3331 / .3331 | `phase2/params/pool_weights.json`, `scripts/constants_2025.sh pool` |

## 6c. Stage-by-stage md5 chain
Reproduced with the code of this repository on the GX10 (aarch64, 4 threads, polars 1.44.1, numpy 2.5.2) from the saved stage inputs of
the two line workspaces (chain production outputs `PR_*_HA0L`, linked fold vectors, Kumo arm files `ARM_F1/F3/F4`, stage slots), each
step on the previous replayed output (v100 on the replayed `final.v99`); the assembly also on an x86_64 machine, twice. Every md5 below
equals the saved output of the build. The steps ran the commands of `post_iso` / `step_post_v99` (`phase2/run_pipeline.sh`) and
`iso` / `isolate` / `step_krs` / `step_krsnl` (`phase2/steps/p2_v107.sh`) with the file names of the build's workspaces; the stages before
the chain outputs (members, Kumo, chain stage refits) were not rerun.

| stage | step (script) | line A md5 | line B md5 |
|---|---|---|---|
| line | `iso_L` (`pool_mix.py` → `pru_daily_calib.py` → `p2_slot.py` → `p2_postrules.py` ×2 → `p2_exposed_scenarios.py`) | `89649021b0f4dcfe0af849520188ad04` | `332052c8441e2f9daa341e416d2755a6` |
| line | `iso_XB` | `5537bd9c28b6064c4fb14a61a3887966` | `f833b7e96e125f0cff2e17d6df4fb79d` |
| line | `iso_XS` | `23d4bc08adb6c03f5c1205b27e6634cb` | `3f28600d793ec8fc67f673a0951e442e` |
| line | compose (`p2_compose.py`) | `bbd07b6a55ed1edc7fe8d55590e02c6a` | `058b423b1397e541afc2404e9838c528` |
| line | E1s (`p2_e1s.py`) | `c7c329a604ffd20a6dd90af6d853602c` | `c93fa629f7dcbe45445ab8e4e1657748` |
| line | Kumo F1 / F3 build (`kumo_icl_build.py`) | `d47e2f89a821856c0c7bf1fb81cce6f9` | `a4c6df011273c9e281186cc18ff90ade` |
| line | Kumo F4 build (`kumo_icl_lirf.py`) = `final.v99` | `f5570dc8dcea40550e646630a0a84ab5` | `bc0fa06831e8660bf3a3d09bc165c100` |
| levers | iso `nbr_clock_twin_bm` / `NBR_CLOCK_TWIN` (`p2_v107.sh` `iso`, the post line of `iso_L` on the variant chains) | `05568d46f8ee2f84e0b02ca5b9667356` / `af837a7c09b250e64bf0bc43c38b66b1` | |
| levers | iso `k250_stack` / `KSTK_CLOCK` | `4e1367bb9b31fe4e0225e869e90b7078` / `b056658645617bded7f4530e31f1a0ef` | |
| levers | iso `BIG_NOVELTY` / `kstk_input` / `kcatlg_stack` | `4ac6acd58116b78518734497dff478dd` / `e31c84b3209090de8a6b53daf80f7515` / `fabb651908d689ae9c7f55ea8d233be7` | |
| levers | v100 → v102 → F2K → v106b → v106K (`isolate_delta.py`) | `28aa67cdede64ade91819ef7be27eaf9` → `24c356a9a3b26c154ae6e062b47f4bac` → `47d8f7a903b1762ac714bc26d0091e7c` → `e32748570467ab557a6f2b9729a816d8` → `d0c4e61c344b44915fe565121328ba8c` | |
| levers | TW1′ + KRS (`krs_build.py`) = `final.v107`; KRS-NL (`krs_nonlinear.py`) = `final.v111` | `e9004bf86c4e63a0bc6cd54ef65c3dec`; **`e435443735aba545bef0499eabacf7c8`** | |
| assemble | F1 → F4 (§6) | `ab87ba63…`, `34c4a6e5…`, **`1a155e9f…`**, `9cd6296d…` | |

What an organiser gets: from the saved stage inputs, the md5s above. From the raw files, the members (§7) are new draws, so the line
outputs and the final file differ from the submitted bytes within the refit noise of §7.

## 7. Determinism
| source | stage | size | from saved stage inputs | from raw files |
|---|---|---|---|---|
| LightGBM member refit, same command and machine | members; the pseudo-label member inside line | a new draw (e.g. `lgb_adsb_linked_a` s0: 6.1 s RMS on the same machine, 24–33 s between seeds); aarch64 and x86_64 differ; classifier probabilities at 4 threads differ in the last digits | not refit: md5s of §6c | new draw |
| CatBoost on GPU | members | not bit-identical between GPU types; a saved model re-predicts bit-exactly | not refit | new draw |
| Kumo GPU fp16 | kumo, kstk | 0.4–1.3 s RMS per instance between GPU types; repeated runs on the same GB10 bit-identical (184 of 184 instances) | not rerun | bit-identical on a GB10 |
| polars multi-threaded float sums in two fold-side group-bys | A side | 1,106 rows on two days, ±3 s | fold vectors are saved inputs | reported, not fixed |
| chain stage refits on the fold vectors (C4, tree D, Hu6, A0 + L) | line, levers | deterministic for a given machine and thread count | saved chain outputs are inputs of §6c | – |

## 8. Checksums of the competition files
| file | rows | md5 |
|---|---|---|
| `final_ranking.parquet` (re-issued 8 Oct 2026 11:04 UTC) | 1,341,197 | `1ce0c6ee9ea15b8c3d0cd282591d4e0a` |
| `final_submitting.parquet` | 670,790 | `2c1f407296e217fd7d242c9d51ba27ad` |
