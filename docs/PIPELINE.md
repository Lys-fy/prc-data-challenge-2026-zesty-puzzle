# Pipeline map

Scripts on the final path only: the stages of `scripts/run_final.sh final` (part 1; line A = `phase2/periods/final.env`, line B =
`phase2/periods/final_bag.env`) and the assembly of the submitted file from the two line outputs (part 2, `scripts/assemble_final.sh`;
`scripts/assemble_final_staged.sh` replays the four-run build with the same result). The 2025 side (member OOF, chain fold sides,
2025 constants) is built once beforehand ([`REPRODUCE.md` §5](REPRODUCE.md#5-a-2025-side-once)) and linked read-only into the workspace.
Paths in the input and output columns are relative to the workspace `phase2/work/final/` unless they start with `scripts/`, `phase2/`
or `data/` of the repository. Runtimes are measured on the final ranking file (670,790 rows, [`REPRODUCE.md` §2](REPRODUCE.md#2-runtimes-measured));
"GPU" means a CUDA GPU with at least 16 GB, "CPU" means 4–6 threads and at most 64 GB of RAM. Code labels: `FORM=v99` and the
`v99_*` files are the base line, `p2_v107.sh` is the chain-variant part, `final.v111.parquet` is the line output before the
schedule-copy rule, and `public_janjul` is the January + July 2026 part of the public ranking file.
Project root: scripts find it in one of three ways: from `PRC_ROOT` (unset: the current directory), from their own location (the
parent of `scripts/`), or as paths relative to the current directory. Run every command from the repository root with `PRC_ROOT` set
to it; inside a workspace the root is the workspace, whose `scripts/` is a copy (`phase2/p2ws.py`).

## 1. Stages: `scripts/run_final.sh final [stage ...]`

| step | script (called by) | inputs | outputs | runtime | host |
|---|---|---|---|---|---|
| 0. 2025 constants check (stage `inputs`) | `scripts/run_final.sh`; `scripts/constants_2025.sh` with `REFIT_CONSTANTS=1` | `phase2/params/postrules_fixed.json`, `pool_weights.json`, `krs_coef.json`, `krsnl_coef.json`, `scratchpad/xkrsnl/prod/gbm_A.txt`, `scratchpad/xhr/unm2025b.parquet` | exit 1 if one is missing | not measured separately | CPU |
| 1a. workspace (stage `inputs`) | `phase2/run_pipeline.sh final ws` → `phase2/p2ws.py` | `phase2/periods/final.env`, `scripts/`, the A-side files | workspace with a byte-identical copy of `scripts/` (`manifest/scripts.md5`), link farms, `logs/ledger.jsonl` | not measured separately | CPU |
| 1b. period inputs | `phase2/steps/p2_final_inputs.sh` (`opdi side1 adsb_extract adsb_tables check`) → `build_opdi_features.py`, `build_xba_features.py`, `build_xe_queue_features.py`, `build_durp_features.py`, `p2_period_inputs.py` | `data/final/final_ranking.parquet`, adsb.lol day files, OPDI, PRU csv, METAR | ADS-B clock side tables (`adsb`, `adsb2`, `adsb3`, `adsb5`, `adsb6`, `adsbL`), OPDI and queue side tables, coverage report | ≈ 13 s per day of clock extraction; with 1c–1d ≈ 1.5 h | CPU |
| 1c. features | `phase2/run_pipeline.sh final np_features` → `scripts/build_features.py --rank` | 2025 training files, final ranking file, open-data tables | `data/features_v9.parquet` (2025 + ranking rows), `dreg` table | in 1b | CPU |
| 1d. ranking-dependent side tables | `run_pipeline.sh final np_side` → `build_dreg_features.py`, `xba_fallback_ids.py`, `fallback_classifier.py`, `queue_regime_prob.py` | features, side tables | fallback probability `q`, delay-regime and queue tables of the ranking rows | in 1b | CPU |
| 1e. fold-side links | `phase2/steps/p2_v107.sh final v107link` | A-side chain fold files, KRS / KRS-NL constants | read-only links in `scratchpad/` | not measured separately | CPU |
| 2. members (stage `members`) | `scripts/xsave/final_refit.sh <family> <seed>` per line of `scripts/xsave/members_xsave.tsv`; NBR_CLOCK MBm / MA (`nbrclock_member.py`); BIG s0–s2 (`dayset_big_member.py`); `scripts/members_check.sh` | features, side tables, 2025 labels | `data/preds/<member>_s<seed>.parquet`, saved models in `artifacts/xsave/` | ≈ 330 CPU-h + ≈ 20 GPU-h | CPU (LightGBM); GPU (CatBoost, RealMLP, BIG) |
| 3. Kumo members (stage `kumo`) | `scripts/kumo_prod.sh final k250stage k250 k250collect kcat` → `kumo_gpu/stage/stage_data.py`, `kumo_gpu/cluster/km_engine.py`, `kumo_gpu/collect/collect.py`, `kumo_kcat_codes.py`, `kumo_kcat_run.sh`, `kumo_kcat_logtarget_run.sh`, `kumo_kcat_logtarget_bag.sh`, `kumo_kcat_logtarget_tailguard.py` | features, 2025 labels (context), Kumo Tabular v1.0.0 weights | K250 (`data/preds/KSBWPc100P_s0.parquet`, instance spread), KCAT8, KCATL16, KCATLg | K250 2.0 h; KCAT8 + KCATL 3–4 h | GPU |
| 4a. unmatched rows and stage inputs (stage `line`) | `phase2/run_pipeline.sh final` steps `pre`, `pre_v87` → `p2_unmatched.py`, `p2_slot.py`, `fallback_assemble.py`, `p2_unm_resid_apply.py`, `unm_link.py`, `unm_link_lever.py`, `p2_unm_link_apply.py`, `calib_unmatched_proxy.py`, `chain_prep.py` | member predictions, features, ADS-B traces | row classes (`scratchpad/p2/U.parquet`), `stages/unmatched_rule.parquet`, `stages/unmatched_resid.parquet`, callsign-linked unmatched rows | ≈ 1.5 h for 4a–4e | CPU |
| 4b. three chains | steps `p13stk_<C>`, `fold_<C>`, `prod_<C>` (C = dTK, Kq, NNA3) → `chain_stk.sh`, `p2_bundle_prod.sh` → `chain_assemble.py`, `combo_c4_apply.py`, `combo_treed_apply.py`, `clockstage_run.py`, `chain_clock_apply.py`, `leave_apply.py` | member predictions, linked fold sides | per-chain production line after meta stacker, C4, tree D, Huber clock stage, leave-row stage | in 4a | CPU |
| 4c. Kumo on unmatched rows | steps `kumo_f13`, `kumo_f4` → `kumo_icl_unmatched.py`, `kumo_icl_lirf.py` | unmatched rows, 2025 unmatched context | residuals F1 (linked other unmatched rows), F3 (unlinked other unmatched rows), F4 (unmatched LIRF rows) | in 4a | CPU (4 threads) |
| 4d. wide-leaf and pseudo-label chains | steps `xb_prod_<C>`, `xs_pseudo`, `m_pseudolabel_pl3p88`, `xs_prod_<C>` → `wideleaf_bundle_prod.sh`, `pseudo_labels.py`, `pseudo_bundle_prod.sh` | first-pass line `submissions/v99_iso_L.parquet` (pseudo-labels), wide-leaf and pseudo-label members | Kq and NNA3 chains with the extra member | in 4a | CPU |
| 4e. pool and post-rules | steps `iso_L`, `iso_XB`, `iso_XS`, `post` → `pool_mix.py`, `pru_daily_calib.py`, `p2_slot.py`, `p2_postrules.py`, `p2_exposed_scenarios.py`, `p2_compose.py`, `p2_e1s.py`, `kumo_icl_build.py`, `p2_check.py` | chain lines, PRU daily file, `phase2/params/postrules_fixed.json` | `submissions/final.v99.parquet` (base line) | in 4a | CPU |
| 5. KSTK (stage `kstk`) | `scripts/kumo_prod.sh final kstkdump kstkbundle kstk kstkgk` → `kstk_run.sh`, `kumo_gpu/cluster/km_adapters_kstk.py`, `kstk_gk.py` | chain-stacker meta frames of Kq and NNA3, A-side KSTK fold runs | `data/preds/KSTK6KqP.npy`, `KSTK6NNA3P.npy` | 2.3 h | GPU |
| 6. chain variants and re-steps (stage `levers`) | `phase2/steps/p2_v107.sh final` → `chainvar_prod.sh`, `kumo_clock_chain.sh`, `dayset_chain.sh`, `kstk_run.sh prod`, `isolate_delta.py`, `p2_novelty.py`, `p2_tw1_rule.py`, `krs_build.py`, `krs_nonlinear.py` | `final.v99.parquet`, K250, KCATLg, BIG, KSTK, NBR_CLOCK, KRS / KRS-NL constants | `submissions/final.v111.parquet` (line output) | ≈ 2 h (one chain variant 15–25 min at 6 threads) | CPU |
| 7. line B (stage `bag`) | copy of `phase2/work/final` to `phase2/work/final_bag` with the base-line markers removed; `phase2/run_pipeline.sh final_bag` (steps 4a–4e with `FZ_L` / `FZ_C`) | line A's workspace after the members, `phase2/periods/final_bag.env` | `phase2/work/final_bag/submissions/final.v99.parquet`, `scratchpad/p2/U.parquet` | ≈ 2.2 h | CPU |
| 8. assembly (stage `assemble`) | `scripts/assemble_final.sh` (part 2) | line A `final.v111.parquet` and `data/features_v9.parquet`, line B `final.v99.parquet` and `U.parquet` | `scratchpad/final_assembly/zesty-puzzle_final.parquet` | ≈ 10 s | CPU |

## 2. Assembly of the submitted file: `scripts/assemble_final.sh`

Inputs: the four files of `manifest/final_inputs.md5` (line A `final.v111` `e4354437…`, line B `final.v99` `bc0fa068…`, line B row
classes `137042cf…`, line A features `c9a079ef…`), checked before the run; outputs checked against `manifest/final_outputs.md5`.
≈ 10 s on 4 CPU threads, byte-identical on aarch64 and x86_64. The md5 chain of every stage is in
[`REPRODUCE.md` §6c](REPRODUCE.md#6c-stage-by-stage-md5-chain).

| step | script | inputs | output |
|---|---|---|---|
| F1 | `scripts/rule_schedule_copy_lirf_relink.py` | `data/final/final_ranking.parquet`, 2025 training files | `schedule_copy/final_dryrun.parquet` (69 rows) |
| F2 | `scripts/rulebag_rows_merge.py` | `kcatlg_line`, `bag_line`, `bag_classes` | `F2_merge.parquet` |
| F3 | `scripts/rule_schedule_copy_lirf_apply.py`, then `scripts/validate_submission.py` | F2, F1, `features_final` | `zesty-puzzle_final.parquet` (md5 `1a155e9fc6e43cff09ca3b9f46b7cede`) |
| F4 | `scripts/public_period_extract.py` | F3, `data/features_v9.parquet` of the public period | `public_janjul.parquet` (md5 `9cd6296dabbf5190e2e79765a897c33b`) |

`scripts/assemble_final_staged.sh` replays the build's staged assembly from four line outputs (K250, KCAT8 and KCATLg in the Kumo-member
slot, plus line B; [`REPRODUCE.md` §6b](REPRODUCE.md#6b-the-staged-assembly-of-the-build)); its two delta steps cancel, so it writes the
same two files.

## 3. Member trainers: `scripts/train2.py` and its seven copies

The LightGBM / CatBoost mixture members are fitted by `scripts/train2.py` and seven copies of it that add flags; the recorded member
commands (`scripts/xsave/members_xsave.tsv`, `phase2/manifest/members.tsv`) name the copy they need. The copies are kept as built:
a merge into one trainer with thin wrappers gave byte-identical outputs on a synthetic one-thread table (17 runs, every flag), but
production refits at 4 threads changed the regressor output of queue_eb_v47lqa (`train2.py`) by up to 87.5 s and of lgb_noclock_nc by
up to 117 s on every ranking row, so it was reverted. `bash tools/trainer_diff.sh [<x>]` prints the diff of each copy against
`train2.py`. Differences (x = present):

| feature of the script | train2 | wideleaf | pseudo | queue | catcfg | linked | clockfree | dark |
|---|---|---|---|---|---|---|---|---|
| `--objective phuber` (pseudo-Huber loss, start score = weighted median) | | x | x | | | | | |
| `--pseudo`, `--pseudo-w` (pseudo-labelled ranking rows) | | | x | | | | | |
| `--xq-branch` (two regime regressors mixed by the queue probability) | | | | x | | | | |
| `--aux-rounds-base`, `--cat-simple-ctr`, `--xh-print-borders` | | | | | x | | | |
| `--reg-wfile`, `--reg-extra`, `--target resid_lobt` | x | x | x | | | | | |
| `--rank-p-only` | x | x | x | x | | x | x | x |
| `--target resid_clock`, `--linked-only`, `--clk-lo/--clk-hi`, `--no-xlk-feats` (clock anchor, xlk features) | | | | | | x | x | x |
| `--dark-only`, `--mask-val-clocks`, feature group `xw2rw` | | | | | | | x | |
| `--only-apt`, `--unlinked-only`, `--only-heavy`, `--apt-weight`, `--fit-linked`, `--target mov\|rep` | | | | | | | | x |
| saved-model hooks `--rank-only`, `--save-model`, `--load-model` (`scripts/xsave_lib.py`) | x | x | | x | x | x | x | x |
| ADS-B side-table directory override | x | x | x | | x | | | |

Members per copy: wideleaf_a2p (wideleaf), pseudolabel_pl3p88 (pseudo), queue_decomp_ebqq (queue), CatBoost configuration members
(catcfg), lgb_adsb_linked_a (linked), lgb_noclock_nc (clockfree), lgb_airport_<APT> (dark); every other LightGBM / CatBoost member
uses `train2.py`.
