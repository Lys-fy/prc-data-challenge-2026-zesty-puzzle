# Short labels

Member names, chain variants and rules keep short labels because they are the keys of the files the pipeline writes
(`data/oof/<member>_<fold>.npy`, `data/preds/<member>_s<seed>.parquet`, chain prefixes in `scratchpad/`). This table maps each one to what it is.

| label | meaning |
|---|---|
| dTK, Kq, NNA3 | the three stacking chains: they read the LightGBM, the CatBoost and the RealMLP member family |
| C1 | unmatched-row baseline blend (V47L + CATV17x) |
| C4 | same-day neighbour correction of the chain (demeaned per airport-day) |
| tree D | nested LightGBM step on the C4 residual |
| Hu6 | Huber LightGBM clock stage on ADS-B-linked rows |
| A0 + L | no clock correction at LIRF; leave-row stage elsewhere |
| (B) | daily calibration on EUROCONTROL's published pre-departure delay |
| V25L, V25Lb, V47L | LightGBM mixture members (L family); V25* form the seed bag of the rule-priced rows |
| CATV17x, CATV9x | CatBoost mixture members |
| XE_*, queue_eb_*, XQ_* | runway-queue-aware members; XQ = queue decomposition |
| lgb_adsb_linked_a | clock-anchored member (ADS-B-linked rows) |
| lgb_noclock_nc | clock-free member (rows without ADS-B link) |
| lgb_airport_<APT> | airport-dedicated member |
| wideleaf_a2p, XB | wide-leaf (1,023 leaves) member and its chain |
| pseudolabel_pl3p88, XS | pseudo-label member (second pass) and its chain |
| NNRA2, meta_oof_realmlp_nnra3 | RealMLP members |
| NBR_CLOCK (MBm / MA variants); chain variants nbr_clock_twin_bm / NBR_CLOCK_TWIN | neighbour-clock member (set stats of neighbours' ADS-B clocks) and its twin without the neighbours' clocks |
| K250 (`KSBWPc100`) | Kumo Tabular member: the airport's whole 2025 pool as context |
| KCAT8, KCATL, KCATL16, KCATLg | Kumo member with 8 categorical columns; its log-target twin; the 16-instance bag of both; the bag with the tail guard |
| KSTK (`gK`) | Kumo Tabular on the chain-stacker frame |
| BIG | airport-day set transformer |
| k250_stack, KSTK_CLOCK, BIG_NOVELTY, kstk_input, kcatlg_stack | chain variants: K250 as stacker input; K250 also in the clock stages; novelty-masked BIG set transformer; KSTK; KCATLg |
| KRS, KRS-NL | Kumo re-step: re-step toward K250 by Kumo-spread tercile; its non-linear form |
| TW1′ | LFPG return-to-stand twin rule |
| standfloor, offblock_pull | stand-occupancy floor; late-NM-off-block pull |
| unm_resid | unmatched-row residual stacker |
| months39, months28, months410, janjul, months612, months511 | the 2025 month-pair folds 3+9, 2+8, 4+10, 1+7 (January + July), 6+12, 5+11 |
| F1, F3, F4 | Kumo in-context learning on linked unmatched, unlinked unmatched and unmatched LIRF rows |
| E1s | own at-stand ADS-B clock rule for exposed non-LIRF rows |
| PX-safe | scenario-safe rule for exposed LIRF rows |
| schedule-copy rule (`rule_schedule_copy_lirf_*.py`) | schedule-copy rule at LIRF (join-rule regime call) |
| `FORM=v99`; `p2_v107.sh` steps `v100`, `v102`, `f2k`, `v106b`, `v106k`, `krs`, `krsnl` | code labels of the two parts of the production line: the base line (`phase2/run_pipeline.sh`) and the chain-variant part (`phase2/steps/p2_v107.sh`): neighbour-clock, Kumo-member, Kumo-in-clock-stage, set-transformer and Kumo-stacker variant steps, then the Kumo re-steps |
| `final.v111.parquet` | the line output of the chain-variant part (`submissions/` of a workspace), before the schedule-copy rule |

## Line version codes

The production line is a sequence of files; each code names the file after one step. Codes appear in file names
(`submissions/final.v99.parquet`, `scratchpad/xpr/v107_<step>.parquet`), step names and `FORM` values. The numbers are labels only.

| code | file / step | content |
|---|---|---|
| v41 | inside the chain stackers (`scripts/chain_stack.py` and its variants) | stacker base `cf(0.5·L + 0.5·C)`: mean of the LightGBM and CatBoost member predictions, LOBT-bounded; the meta stacker fits `y − v41` |
| v53, v56 | `--v53 stages/unmatched_rule.parquet`; comments of `scripts/*_stack.py` | the unmatched-row stage file: rows equal to it are the protected rows the matched-row stages must not change; also names the default meta-stacker member set RA3 / RAC2 / NNA2 |
| v62, v98 | `scripts/exposed_characterise.py`, `scripts/kumo_icl_lirf.py` | a saved line file that these research scripts read as the incumbent values of the unmatched-LIRF and exposed rows; v98 carries v62's values on those rows |
| v76 seeds | `phase2/manifest/members.tsv`, column 2 | the seed range of each member family retrained by `step_member` |
| v78 | `FORM=t2x` of `phase2/run_pipeline.sh` | chains with the `KB` stacker inputs + `xpr` + `A1`, fallback rows taken from the member |
| v86 | fold line `scratchpad/xau1/clockfold1_le3ba_<fold>.npy` | chain fold line with (B) and the monthly PRU add-on (`LE3BA`); its (B)-only twin `clockfold1_le3b` is the fold teacher of the pseudo-labels (`scripts/pseudo_labels.py`) |
| v83 | clock-stage inputs (`phase2/steps/p2_period_inputs.py`) | clock stages that read the adsb6 push-start clock (`scripts/adsb_pushstart_extract.py`) |
| v87 | `FORM=v87` | the base chains: `KB` fold stacks, `CLOCKFOLD1` clock stage, `SIXSEED` production; unmatched rows `U3`; with the new-stand delta |
| v88 | `post_iso` of `run_pipeline.sh`, `I(X)` of `p2_v107.sh` | the v87 post line without the new-stand delta: `w5` pool → (B) → `U3` stitch → standfloor → offblock_pull → PX-safe; `submissions/v99_iso_L.parquet` |
| v89, v91 | `submissions/v99_iso_XB.parquet`, `v99_iso_XS.parquet` | the v88 post line with the wide-leaf (`XB`) or pseudo-label (`XS`) Kq / NNA3 chains |
| v94 | `phase2/steps/p2_compose.py` | `round(v88 + (v89 − v88) + (v91 − v88))`, floor 60 s |
| v99 | `FORM=v99`; `submissions/final.v99.parquet` | base line of the final period: v94 → E1s → F1 / F3 → F4 |
| v100 | step `v100` of `p2_v107.sh` | v99 + isolate delta of the neighbour-clock chains (`nbr_clock_twin_bm` − `NBR_CLOCK_TWIN`) |
| v102 | step `v102` | v100 + isolate delta of the KCATLg chains (`kcatlg_stack` − `XB`); `V102ARM=k250_stack` puts K250 in this slot |
| F2K | step `f2k` | v102 + isolate delta of K250 in the clock stages (`KSTK_CLOCK` − `k250_stack`) |
| v106b | step `v106b` | F2K + isolate delta of the BIG set-transformer chains (`BIG_NOVELTY` − `k250_stack`) |
| v103 | comments of `scripts/krs_build.py` | a matched-row line without any KRS step, the input form of `krs_build.py`; in the final line this input is v106K |
| v106K | step `v106k` | v106b + isolate delta of the Kumo-stacker chains (`kstk_input` − `k250_stack`) |
| v107 | step `krs`; `submissions/final.v107.parquet` | v106K with TW1′ (+1,264 s on the LFPG twin rows) and KRS on every other matched, unprotected row |
| v111 | step `krsnl`; `submissions/final.v111.parquet` | v106K with TW1′ and KRS-NL (`scripts/krs_nonlinear.py` arm A); the line output before the schedule-copy rule |

`FORM` values: `run_pipeline.sh` takes `px` (pooled chains with `xpr` inputs, post-rules fitted at run time), `t2x` (v78), `v87` and
`v99` (the final form); `p2_v107.sh` documents its own part as `FORM=v107`. `FORM=PB | PC` of `scripts/chain_stage.sh` and
`scripts/clockstage_fold.sh` is a different switch: `PB` = jittered Huber clock stage, `PC` = `PB` + adsb5 clocks, 255 leaves, 600 rounds.

## Stage, file and run codes

| code | meaning |
|---|---|
| C1 | unmatched-row baseline: V47L + CATV17x blend (`phase2/steps/p2_slot.py --unmatched U`) |
| C4 | same-day neighbour correction (`scripts/combo_c4_apply.py`); defined in [`METHOD.md` §4](METHOD.md#4-stacking-chains) |
| tree D | LightGBM on the C4 residual (`scripts/combo_treed_apply.py`); file suffix `C4treeD`, `TD` |
| Hu6 | Huber LightGBM clock stage, 6 seeds (production `scripts/chain_clock_apply.py` via `scripts/clockstage_run.py`, fold side `scripts/chain_clock.py`) |
| A0 + L | LIRF rows keep the pre-clock-stage value (A0); LightGBM leave-clock stage on linked rows elsewhere (L) (`scripts/leave_apply.py`, `scripts/stage12_a0l.py`) |
| HA0L | file suffix of a chain's final vector: after Hu6 (H) and A0 + L |
| A1 | (1) env `A1=1`: LIRF rows out of the clock-stage fits; (2) callsign-link arm `k 0.5, clip ±600 s` of `scripts/unm_link_lever.py` (unmatched rows `UA1`) |
| V3 | callsign-link arm `k 1.0, clip ±1,200 s` (same g as A1) |
| U, U2, UA1, U3 | unmatched-row value sets in `scratchpad/p2/`: U = C1 / seed-bag rules; U2 = U + unm_resid; UA1 = U2 + link arm A1; U3 = U2 + link arm V3 |
| U3 stitch | `phase2/steps/p2_slot.py --unmatched U3 --matched <pool>`: matched rows from the calibrated pool, unmatched rows from U3 |
| w5 pool | `scripts/pool_mix.py` with five slots `W=0,.3212,.3065,0,.3722` (dTK, dTK, Kq, dTK, NNA3): the three-chain pool written as a 5-vector dot product; .3722 is the 4-decimal NNA3 fit (0.372242, `phase2/params/pool_weights.json`), while the 3-weight lists of `pru_daily_calib.py` use .3723 so that the weights sum to 1 |
| P13 | file prefix of chain fold stacks (`scratchpad/p13/P13_<arm>{STK,TD}_<fold>.npy`, `scripts/chain_stk.sh`, and `P13_XB*`, `P13_XS*`, `P13_<prefix>*` of the variant chains); in code comments it marks the stacker form with the PRU daily inputs (`xpr`) |
| KB | (1) arm prefix of the base chain stacks `KBdTK`, `KBKq`, `KBNNA3`: chain stacker with the member spread (`spr`) and the 10 raw feature inputs (`raw`); (2) in `scripts/kstk_stack.py`: the stacker correction `(1 − W)·gM + W·gK`, W 0.3 |
| RA3, RAC2, NNA2 | default fold-OOF prefixes of the LightGBM (L), CatBoost (C) and RealMLP (N) inputs of the chain meta stacker (env OL / OC / ON); the production chains set OL=queue_eb_ra3qa, OC=meta_oof_h_ctl, ON=NNA2 (`phase2/run_pipeline.sh` `chain_env`) |
| gK, gM | KSTK correction `gK = Kumo taxi − v41`; the chain meta stacker's own correction `gM` |
| KSTK6 | run name of the Kumo chain-stacker instances (6 chain folds + ranking rows): `data/oof/KSTK6<chain>_<fold>.npy`, `data/preds/KSTK6<chain>P.npy` (`scripts/kstk_gk.py`) |
| KSBWPc100 | Kumo run profile of K250 (`kumo_gpu/acceptance/sb_profiles.txt`): medium model, whole 2025 pool of the airport as context (cap 1,000,000 rows), 100 columns by importance, 8 instances |
| AMXAU1 | ARM tag of the base-chain production files `scratchpad/xau1/PR_L<C>_AMXAU1_*.parquet`: six-seed pre / C4 / tree D (AM) with the CLOCKFOLD1 clock stage (XAU1); `XBAU1`, `XSAU1`, `XKAU1` = the same for the XB, XS and variant chains |
| CLOCKFOLD1 | clock-stage form with the adsb5 push-back, adsb6 push-start and adsbL longest-stop clocks (`scripts/clockstage_fold.sh`; files `scratchpad/xau1/clockfold1_*`) |
| SIXSEED | production form of the v87 chains: 6-seed members (`sixseed_chain_ebdt6`, `sixseed_chain_nnra3x6`), 6 meta-stacker seeds, 6 clock-stage seeds |
| LE3BA, LE3B | 2025 fold lines `clockfold1_le3ba` ((B) + monthly PRU add-on) and `clockfold1_le3b` ((B) only) |
| xtd, xtd2, xtd3 | first, second, third extra member input of a chain stacker (`scripts/wideleaf_stack.py`, `chainvar_stack.py`, `dayset_stack.py`): `T = pL·artL + (1 − pL)·taxi_M` and `T − L` for the extra member M; NaN where M has no row |
| XB, XS | chain prefixes of the wide-leaf chains (member `wideleaf_a2p` as `xtd`) and of the pseudo-label chains (L member `pseudolabel_pl3p88`) |
| FZ, FZ_L / FZ_C | seed bag of the rule-priced rows: `FZ = 0.5·mean(FZ_L) + 0.5·mean(FZ_C)`, FZ_L = 24 LightGBM seeds (V25L s0–15, V25Lb s10–17), FZ_C = 16 CatBoost seeds (CATV9x s0–15), listed in `phase2/periods/*.env` |

## Terms

| term | meaning |
|---|---|
| art | the fallback value of the mixture: the taxi time implied when the block stamp is a fallback, `MVT − SCHED` (the scheduled stamp); on unmatched rows with `MVT − SCHED > 8 h` it is mixed with `86,400 s + day-shift offset` by the day-shift probability (`scripts/train2.py`) |
| stake | sample weight of the fallback classifier: the cost of a misclassification, `(max(MVT − SCHED, 0)/3,600)^k + floor` (`scripts/train2.py --cls-weight stake`; defaults k 2, floor 0.05; production, `scripts/xsave/members_xsave.tsv`: `--stake-power 2.42` in every LightGBM / CatBoost family; floor 0.05 (default) in V25L, V25Lb, CATV9x, CATV17x, runway_queue_catv17xq; `--stake-floor 0.5` in V47L and the other LightGBM families) |
| dark rows | departures without an ADS-B link (`adsb_linked` null or 0); priced by the clock-free branches (`lgb_noclock_nc`, stacker input `xdk`) |
| isolate delta | `base + (arm − twin)` on matched rows, where arm and twin are two full post lines that differ only in one chain input; unmatched and exposed rows are protected, then the LOBT bound, floor 60 s and rounding (`scripts/isolate_delta.py`) |
| lever | one step applied on top of the previous line output: an isolate delta, TW1′, KRS or KRS-NL |
| rung | one stage output of an ordered line (pool → (B) → stitch → post-rules); "the (B) rung" is the vector after (B) |
| protected rows | unmatched and exposed ranking rows; every matched-row step asserts they are unchanged |
| A side, B side | the two parts of a reproduction (`docs/REPRODUCE.md`): A = the one-off 2025 side (member OOF, chain fold sides, 2025 constants); B = the production run on a ranking file |
| line A, line B | the two production line runs on the final ranking file: line A = the KCATLg line (`phase2/periods/final.env`, output `final.v111.parquet`) for the matched and other unmatched rows; line B = the bag line (`phase2/periods/final_bag.env`, output `final.v99.parquet`) for the unmatched-LIRF and exposed rows |
| arm | one candidate variant in a screen or gate (a member, a step or a rule, against its twin or placebo); `ARM` / `V102ARM` select the variant in code |
| arm M | the Kumo Tabular member form of the Kumo screen: Kumo on the raw feature columns (`kumo_gpu/data/km_features_412.txt`) with 2025 labels as context, fold OOF (`scripts/kumo_km_member.py`; profiles m20e8 … of `kumo_gpu/jobs/config.env`); K250 (`KSBWPc100`) is this member form with the whole 2025 pool of the airport as context |

## `scratchpad/x*` directories

Working directories under the project root (or a workspace root); none is tracked. Writer = the script that creates the files.

| directory | writer | content |
|---|---|---|
| `x4` | `scripts/postrule_fold_frame.py` | `X4_fold.parquet`: 2025 fold rows with their ranking-mix weights (post-rule fit) |
| `xatlas` | `scripts/error_atlas_line.sh` | `rows_matched.parquet`: per-row 2025 fold line (error tables, TW1′ fit) |
| `xau1` | `scripts/clockstage_fold.sh`, `phase2/steps/p2_bundle_prod.sh` | CLOCKFOLD1 fold vectors `clockfold1_<chain>_*` and base-chain production files `PR_*_AMXAU1_*` |
| `xau2` | `scripts/newstand_map.py`, `scripts/newstand_members.sh` | new-stand mapping and delta members (only with `NEWSTAND_DELTA=1`; off in the final period) |
| `xau3` | `scripts/calib_unmatched_proxy.py` | `um25_A1.npz`: 2025 unmatched-row values for the (B) day sums |
| `xbase` | `scripts/wideleaf_stk.sh`, `scripts/wideleaf_bundle_prod.sh` | XB fold stacks `P13_XB*` and production files `PR_XB*_XBAU1_*` |
| `xbig2` | `scripts/tw1_twins.py` | `twins.parquet`: LFPG same-callsign departure pairs |
| `xchain` | `scripts/kumo_clock_chain.sh` | intermediates of the KSTK_CLOCK chains |
| `xchain2` | `scripts/krs_fold_line.sh` (`scripts/krs_fit2.py`) | KRS coefficients `v106K_K2_KRS_prod_coef.json` |
| `xclu`, `xkfm`, `xcluf` | `scripts/kumo_env.sh`, `kumo_gpu/stage/stage_data.py`, `scripts/kumo_kcat_run.sh` | Kumo payloads, run directories and memory logs |
| `xcpu2` | the k250_stack fold job (2025 side; linked into a workspace by `p2_v107.sh` step `v107link`) | k250_stack fold stacks `P13_XKS*` and the (B) fold line `chain_fold_line_xks_w5_le3b` |
| `xcr`, `xz` | none | extra search paths for fold vectors of `combo_c4_apply.py`, `combo_treed_apply.py`, `stage12_clock_apply.py`, `chain_clock_apply.py` |
| `xds2` | `scripts/dayset_novelty.py`, `phase2/steps/p2_novelty.py` | BIG novelty days and masks |
| `xfinal` | `phase2/steps/p2_final_inputs.sh`, `p2_v107.sh` | final-period ADS-B, OPDI and METAR extraction inputs; TW1′ ids `tw1p_ids.parquet` |
| `xhook` | `scripts/chainvar_*.sh`, `dayset_*.sh`, `kstk_*.sh`, `kumo_clock_chain.sh` | variant-chain fold stacks `P13_<prefix>*` and production files `PR_<prefix><C>_XKAU1_*` |
| `xhr` | `scripts/exposed_*.py`, `p2_v107.sh` step `exposed` | exposed-row ids, PX-safe bounds `exposed_ord_bounds.json`, scenario ledgers, 2025 unmatched frames `unm2025*.parquet` |
| `xhr2` | `scripts/rulebag_blend.py` | `fzr_rank.parquet`: FZ seed-bag blend on the ranking rows |
| `xkrsnl` | `scripts/krs_nonlinear.py` | KRS-NL fold vectors and the production fit `prod/coef.json`, `prod/gbm_A.txt` |
| `xkstk` | `scripts/kstk_run.sh` | KSTK collection report `gk.txt` and gate outputs |
| `xkt`, `xpla`, `xcap` | `kumo_gpu/collect/collect.py`, `scripts/screen_l2_extra.py`, `scripts/kumo_t2x_screen2.py` | Kumo member screens (research) |
| `xkumo3` | `scripts/kumo_kcat_codes.py` | categorical codes of KCAT8 / KCATL |
| `xl` | `scripts/daystate_lib.py` | `frame.parquet`: day-state frame |
| `xlv` | `scripts/chain_leave.py`, `scripts/leave_fit_tuned.py` | leave-stage fold vectors |
| `xmonths` | `scripts/months_rows_table.py` | `rows.parquet`: per-month fold rows of the final line |
| `xnclk` | `scripts/nbrclock_member.py` | NBR_CLOCK member outputs |
| `xof` | `scripts/stage12_mkstk.py`, `run_pipeline.sh` step `fold_<C>` | `oof_build_<name>*` fold vectors of the `px` chain form |
| `xpb` | `scripts/chain_jobs/clockstage/j_L.sh` | push-start comparison fold vectors |
| `xpr` | `phase2/run_pipeline.sh`, `phase2/steps/p2_v107.sh` | production-line intermediates: pool, (B), post-rule files, `v99_*` and `v107_*` step outputs; (B) fold vectors `pru_pxxb_*` |
| `xquick` | `scripts/tw1_rule_build.py` | `tw1p_shift.json` (TW1′ shift), `tw1p_ids_2026.parquet` (public-period twin ids) |
| `xsave` | `scripts/xsave/final_refit.sh` | member refit log `final_fit.jsonl` |
| `xsyn` | `scripts/pseudo_labels.py`, `scripts/pseudo_*.sh` | pseudo-label files and XS chain files |
| `xt` | `scripts/chain_prep.py` | `prep_<fold>.parquet`: fold-side day-state columns |
| `xtfm2` | `scripts/kumo_icl_unmatched.py`, `scripts/kumo_icl_lirf.py` | Kumo F1 / F3 / F4 arm files `ARM_*.npz` |
| `xu` | `scripts/pru_assemble.py`, `scripts/audited_assemble.py` | `XU_<TAG>_pre.parquet`; unmatched reference vectors `unm_resid2_blkb_<fold>.npy` |
| `xur` | `scripts/unm_resid_d.py`, `scripts/unm_resid_s.py` | unm_resid models and vectors `D_*.npz`, `S_*.npz` |
| `xur2` | `scripts/unm_link*.py` | callsign-link frames, IATA→ICAO map, arms `ARM_A1.npz`, `ARM_V3.npz` |
