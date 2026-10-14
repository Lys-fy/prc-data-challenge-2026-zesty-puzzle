# Method

Target: taxi-out time `TAXITIME_SEC_mvt` = take-off (`MVT_TIME`) − off-block (`BLOCK_TIME`), in seconds, for every departure at EDDF,
EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM. Training data: the 2,085,047 departures of 2025. Metric: RMSE. Prediction sets:
the public ranking file (Jan + Jul 2026, 344,841 departures) and the final ranking file (Jan + Feb + Jun + Jul 2026, 670,790 departures).
The same code runs on any ranking file (`phase2/periods/<period>.env`).

**In plain language.** Every model predicts how far the taxi-out time is from a second, independent off-block clock (the Network
Manager's `AOBT_3`), not the taxi-out time itself. Gradient-boosting, neural and tabular foundation-model members make that prediction;
three stacking chains combine them; a daily calibration to published delay data, three post-rules and two re-steps toward the
foundation model where its instances agree follow. The 1.08 % (22,424) of 2025 departures without a flight-plan match are
priced by fixed rules that read no 2026 label. Fitting uses the 2025 labels, label-free statistics over the ranking rows, the published
daily PRU delay aggregates of the scored days ((B) and the `xpr` inputs; no taxi-time label) and, for
the pseudo-label member, the run's own first-pass predictions on the ranking rows (§9). Five components (E1s, the own at-stand clock
rule for exposed rows; the wide-leaf twin; the pseudo-label member; KRS-NL, the non-linear Kumo re-step; KCAT8, the raw-target Kumo member with
8 categorical columns, which KCATLg uses on the rows it predicts above 2,400 s) and the combinations of uploads 69, 85 and 91 were kept on the public score
(§7, [`upload_log.md`](upload_log.md)). Short labels such as K250, KRS or TW1′ are
defined where they first appear and listed in [`naming.md`](naming.md).

## 1. Data and the three facts that shape the model
1. **The Network Manager off-block `AOBT_3` is the anchor.** `MVT − AOBT_3` alone scores 385 s RMSE on 2025. It is a different sensor
   from the airport block time (median offset +51 s, +296 s at LTFM), so every model predicts the residual `y − (MVT − AOBT_3)`.
2. **The tail decides the score.** Labels over 2 h are 0.03 % of the rows and carry 46 % of the squared deviation. Most are departures
   whose block column holds a fallback value: the scheduled time, a clock on the wrong date, or a real long delay. These rows are scored as
   exported (organiser statement), so they are modelled, not filtered.
3. **Open ADS-B gives a third clock.** adsb.lol traces of the departure's own airframe show when it actually started to move.

Unmatched and exposed rows are 1.08 % of 2025 (22,424); the 17,569 outside the sealed days are 0.85 % of the 2,080,192 scored rows and carry 44 % of the out-of-fold squared error ([`figures/row_types.png`](figures/row_types.png)).
Row classes are defined by label-free rules and recomputed for every ranking file:

| class | rule | path |
|---|---|---|
| matched | the flight-plan join exists | members → chains → levers (§3–§5) |
| `unm_other` | no flight-plan match, not LIRF, `MVT − SCHED ≤ 8 h` | unmatched-row path (§6) |
| `unm_LIRF` | no flight-plan match at LIRF, `MVT − SCHED ≤ 8 h` | rule-priced rows (§6) |
| exposed | no flight-plan match, `MVT − SCHED > 8 h` | rule-priced rows (§6) |

## 2. Features
One code path (`scripts/build_features.py`) builds `features_v9.parquet` for the 2025 rows and the ranking rows together; per-row side
tables come from `scripts/build_*_features.py`.

| group | content | source |
|---|---|---|
| time deltas | `MVT` − `AOBT_3`, `EOBT_1`, `LOBT`, `IOBT`, `SCHED` | competition |
| `tl`, `turn` | runway timeline; turnaround (inbound block-in, stand occupancy) | competition |
| `ref`, `te`, `te2` | stand × runway references; out-of-fold target encodings | competition |
| `nbr`, `nbrx`, `arrx`, `arrfb` | neighbouring departures and arrivals, forward and backward windows | competition |
| `durp`, `dreg` | pre-departure plan changes; label-free airport delay regime | competition |
| `xe`, `xba` | runway-queue state; flight memory (same flight number on earlier days) | competition |
| `wx` | hourly METAR | IEM ASOS |
| `opdi`, `opdir` | OPDI ADS-B milestones and the inbound-leg link | EUROCONTROL OPDI |
| `xpr` | daily PRU indicators: pre-departure delay, traffic, ATFM slot adherence, arrival ATFM delay by cause | EUROCONTROL PRU |
| `adsb`, `adsb2`, `adsb3` | own-airframe ground clocks: first seen, first move, hold end, take-off; relative to learned stand positions; segment clocks | adsb.lol |
| `adsb5`, `adsb6`, `adsbL` | push-back; push-start with heading semantics; longest stop and the class of the trace start | adsb.lol |

ADS-B link: the adsb.lol daily archives are cut to boxes around the ten airports, and each departure is linked to its airframe's trace
by callsign and take-off time. Coverage is uneven (share of linked rows in the final months: LFPG 0.14, LIRF 0.30, LTFM 0.00, others
0.62–0.96); rows without a link use clock-free branches. Forward-looking features over other rows of the provided data are allowed by the
organisers ("what is occupied, what is coming").

## 3. Members
Every member except the in-context learners has a mixture form. A classifier gives the probability `p` that the block stamp is a
fallback (weighted by stake); regressors predict the ordinary taxi and the fallback value `art`; the output is `p·art + (1 − p)·taxi`,
bounded by `|y − (MVT − LOBT)| ≤ 3,606 s` (true on every 2025 row) and floored at 60 s. Production fits use all 2025 rows, with one exception (the pseudo-label member below); the 6-fold
month OOF of §7 feeds every later fit. Recipes: `phase2/manifest/members.tsv`, `scripts/xsave/members_xsave.tsv`.

| member | model | seeds | role |
|---|---|---|---|
| queue_eb_v47lqa, runway_queue_v47lq, queue_decomp_ebqq | LightGBM | 6 + 6 + 6 | main members: runway-queue-aware, queue decomposition |
| V47L, CATV17x | LightGBM, CatBoost (GPU) | 10, 20 | unmatched-row baseline C1; CatBoost family of the chains |
| V25L, V25Lb, CATV9x | LightGBM, CatBoost | 16 + 8, 16 | seed bag of the rule-priced rows (§6) |
| runway_queue_catv17xq | CatBoost | 6 | runway-queue-aware CatBoost |
| lgb_adsb_linked_a, lgb_noclock_nc, lgb_airport_ltfm, lgb_airport_lfpg | LightGBM | 3, 6, 6, 6 | clock-anchored (linked rows), clock-free (dark rows), airport-dedicated |
| wideleaf_a2p | LightGBM, 1,023 leaves | 3 | extra stacker input |
| pseudolabel_pl3p88 | LightGBM | 2 | retrained with the period's matched ranking rows pseudo-labelled by the period's own first-pass prediction (no labels) |
| NNRA2, meta_oof_realmlp_nnra3 | RealMLP (pytabkit) | 2 + 6 | neural family |
| NBR_CLOCK MBm / MA | LightGBM | 1 + 1 | set statistics of the neighbours' ADS-B clocks (MBm) and its twin without them (MA) |
| **K250** | Kumo Tabular, medium, in context | 8 instances per airport | context = the airport's whole 2025 pool; readout = mean of the 999 output quantiles; the spread across instances is kept |
| **KCATLg** | Kumo Tabular, log target, tail guard | 16 + 8 instances per airport | K250's recipe with 8 raw identity columns as categoricals; see below |
| **BIG** | airport-day set transformer (d 192, 4 layers, 6 heads) | 3 | one sequence = all departures of an airport and UTC day, full self-attention; input masked on novelty days |
| KSTK `gK` | Kumo Tabular on the chain stacker's meta frame | 8 instances per airport and chain | K250 as a stacking engine |

**The pseudo-label member (`pseudolabel_pl3p88`).** This is the one member that is trained on rows of the ranking file. Each production
run first builds the line without it. It then retrains the member (the recipe of `queue_eb_ra3qa`, 2 seeds) on all 2025 rows plus the
matched ranking rows of the file being predicted. Those ranking rows carry pseudo-labels: the first-pass prediction of the same run,
not a label. They have weight 0.3 and are kept only where the fallback probability of the production member is below 0.05 and the
first-pass prediction is below 7,200 s. The member feeds two extra chains (Kq and NNA3 with the pseudo-labelled member in place of the
LightGBM member), which enter the line as an isolate delta like the wide-leaf chains. On 2025 the same construction is run per fold:
the held-out months' matched rows are pseudo-labelled by the chain's out-of-fold prediction (`scripts/pseudo_labels.py`,
`scripts/pseudo_member.sh`). In the final run the pseudo-labelled rows are the matched rows of all four months (§9).

**KCATLg, the log-target Kumo member with tail guard** (`scripts/kumo_kcat_*`):
- KCAT8 = K250's recipe plus the raw columns `STAND_mvt`, `RUNWAY_mvt`, `AIRCRAFT_OPERATOR_flt`, `AIRCRAFT_TYPE_flt`, `WK_TBL_CAT_flt`,
  `ADES_flt`, `MARKET_SEGMENT_flt` and `FLIGHT_TYPE_flt`, passed as Kumo categoricals (codes = ranks in the airport's sorted value set, null
  its own code, plus a frequency column for columns with ≥ 50 categories). 8 instances per airport, with the contexts and column views of K250.
- KCATL = the same 8 instances with target `log1p(y)`. The point prediction is the mean of `expm1(q)` over the 999 predicted quantiles, with
  `q` clipped to the context's log-target range: the distributional mean, which needs no separate back-transform correction.
- KCATL16 = the engine aggregation (trimmed mean 0.2) over the 16 instances KCAT8 0–7 and KCATL 0–7.
- Tail guard: a log target compresses the extreme tail, so per row the KCAT8 prediction is used where it is above **T = 2,400 s**, and
  KCATL16 elsewhere (0.93 % of the final rows). T is one global value from the grid {1,800, 2,400, 3,600 s, q99}; the rule is label-free.

BIG novelty days: an airport-day whose label-free day profile lies beyond the 2025 99th percentile of its airport (Mahalanobis distance on
the top-20 principal components, `scripts/dayset_novelty.py`).

## 4. Stacking chains
Three chains, **dTK**, **Kq** and **NNA3**, differ in the LightGBM / CatBoost / neural member they read. Each chain applies:

| stage | model | inputs / scope |
|---|---|---|
| meta stacker | LightGBM, 6 meta seeds, correction clipped to ±150 s | member outputs, fallback probability `q`, ADS-B link flag, member spread, top-10 raw features, daily PRU indicators |
| C4 | same-day neighbour correction, demeaned per airport-day, × (1 − p) | the day's other predictions |
| tree D | nested leave-one-fold-out LightGBM on the C4 residual | |
| Huber clock stage (Hu6) | Huber LightGBM (255 leaves, 600 rounds, 6 seeds, clip ±300 s) | ADS-B-linked rows; adsb5 / adsb6 / adsbL clocks and their gaps to the prediction; LIRF excluded |
| A0 + L | no clock correction at LIRF; leave-row stage elsewhere | |

Every stage is fitted on the 2025 month-fold OOF; production applies the same fitted form to the ranking rows.

**What each matched-row stage computes.** Every stage changes matched rows only; unmatched and exposed rows pass through unchanged.
- **C4** (`scripts/combo_c4_apply.py`). For each matched row, 11 neighbour-state columns: means over the departures of the same airport
  and runway within 5 or 15 min of its take-off of their prediction, member-taxi and reference gaps (to `AOBT_3` and to the stand × runway
  median). The columns are demeaned per airport-day, a least-squares β per airport turns them into a correction, and the correction is
  multiplied by `1 − p`, so the stage moves a row with its neighbours' shared error without moving the day's mean.
- **tree D** (`scripts/combo_treed_apply.py`). A LightGBM fitted to the residual left after C4, on the 11 C4 columns, the link flag,
  `p`, the LightGBM–CatBoost disagreement, the row's own gaps to `AOBT_3` and to the reference, and the airport. Its output is clipped
  to ±150 s and demeaned per airport-day; EGLL and EHAM rows are left unchanged.
- **Hu6** (`scripts/chain_clock_apply.py`, run by `scripts/clockstage_run.py` from `phase2/steps/p2_bundle_prod.sh`). A Huber-loss LightGBM (6 seeds) fitted to `clip(y − line, ±1,800 s)` on ADS-B-linked
  matched rows from the own-airframe clocks (adsb5 push-back, adsb6 push-start, adsbL longest stop) and their gaps to the prediction; it
  adds `clip(g, ±300 s)` to linked matched rows outside LIRF. Rows without a link keep their value.
- **A0 + L** (`scripts/leave_apply.py`). A0: LIRF rows keep their pre-clock-stage value. L: a second LightGBM (400 rounds, 6 seeds in
  production (`MS6=1`), clip ±300 s) on the linked matched rows whose first valid clock is the stand-leave clock (`adsb2_leave`), fitted to the residual after Hu6.
- **Isolate delta** (`scripts/isolate_delta.py`). Each chain variant and its twin are run through the full post line (pool → (B) →
  unmatched stitch → standfloor → offblock_pull → PX-safe); the variant's change over the twin, `arm − twin`, is added to the current
  line on matched rows, then the LOBT bound, rounding and the 60 s floor are applied. The variant, its twin and the step that adds it
  are listed in the table below.

**Chain variants.** The production line has two parts: the base line (`phase2/run_pipeline.sh`, period setting `FORM=v99`, the code label of the base line), and the
chain-variant part (`phase2/steps/p2_v107.sh`). Each variant reruns the whole chain with one extra stacker input. It enters the file as an isolate delta on matched
rows, measured against a same-recipe twin without the new input (`scripts/isolate_delta.py`); unmatched and exposed rows are protected.

| step | variant (Kq, NNA3) | extra input | twin |
|---|---|---|---|
| 1. neighbour clocks | nbr_clock_twin_bm | NBR_CLOCK MBm | NBR_CLOCK_TWIN |
| 2. Kumo member | kcatlg_stack | KCATLg | XB (1,023-leaf chain) |
| 3. Kumo in the clock stages | KSTK_CLOCK | K250 also in the Huber clock stage and A0 + L | k250_stack (K250 as stacker input) |
| 4. set transformer | BIG_NOVELTY | BIG, novelty-masked | k250_stack |
| 5. Kumo stacker | kstk_input | KSTK `gK` | k250_stack |

Order: base line → steps 1 to 5, each on the previous output, then TW1′, KRS and KRS-NL (§5). Step 1 adds the effect of the
neighbours' ADS-B clocks; steps 2 to 5 add KCATLg as a stacker input, K250 inside the Huber clock stage and A0 + L, the BIG set
transformer, and the KSTK correction, each measured against a twin that already holds K250 (or the wide-leaf input) in the same slot. Code labels of the steps:
`v100`, `v102`, `f2k`, `v106b`, `v106k` (`phase2/steps/p2_v107.sh`).

## 5. Pool, calibration, post-rules, re-steps
| step | what | fitted on |
|---|---|---|
| pool | `.3212·dTK + .3065·Kq + .3722·NNA3` | global simplex fit on the 6 folds of 2025, matched rows (`scripts/pool_weights_fit.py`) |
| (B) | soft daily calibration on all matched rows: each airport-day's mean prediction is pulled toward the level implied by EUROCONTROL's published daily pre-departure delay; zero shift on days without PRU data | nested by month on 2025 |
| standfloor | stand floor: if the last arrival on the same stand blocked in within 15 min after the predicted off-block, the prediction is raised | per-airport arms, 2025 folds (`phase2/params/postrules_fixed.json`) |
| offblock_pull | if `AOBT_3 − LOBT > 2 h`, pull toward `MVT − LOBT` | same |
| TW1′ (LFPG twin rule) | LFPG writes a return-to-stand as two departure records of one callsign (`|ΔMVT| ≤ 300 s`). In a pair of exactly two records with exactly one NM-matched, the matched record gets +1,264 s. Record order is not used | shrunk 2025 class mean `n/(n+5)·mean(y − pred)` |
| KRS (Kumo re-step) | every matched, unprotected row: `line + β_k·(K250 − line)`, k = tercile of Kumo's instance spread, β = .4520 / .1726 / .0674: a large step where the instances agree, a small one where they disagree | 2025 fold rows |
| KRS-NL (non-linear Kumo re-step) | `+ 0.5·ĝ`, ĝ = LightGBM fitted to the clipped (±1,800 s) KRS-line residual on [line, K250 − line, spread, airport, UTC hour]; 63 leaves, ≥ 2,000 rows per leaf, 300 rounds; TW1′ rows excluded | all 2025 fold rows |

In plain terms, on which rows each of these acts:
- **(B)** (`scripts/pru_daily_calib.py`) acts on matched rows. For each airport-day it solves for the shift δ (within ±600 s) at which
  the off-block delays implied by the predictions (`take-off − scheduled time − prediction`, smoothed by the airport's residual spread;
  unmatched rows enter at their rule values) reproduce EUROCONTROL's published mean pre-departure delay of that day. Each matched row then
  moves by `k·δ·q`, where q is the row's probability of being delayed and k ∈ [0, 1.5] is fitted per airport and delayed / not-delayed
  class, nested by month on 2025.
- **TW1′** (`scripts/tw1_rule_build.py`, `phase2/steps/p2_tw1_rule.py`) acts on LFPG rows only: when one callsign has exactly two
  departure records within 300 s of each other and exactly one of them is NM-matched, that matched record gets +1,264 s (a return to
  stand logged as two departures).
- **KRS** (`scripts/krs_build.py`) acts on every matched, unprotected row except the TW1′ rows: it moves the line toward the K250
  prediction by a fraction β that depends only on the tercile of the spread across K250's 8 instances.
- **KRS-NL** (`scripts/krs_nonlinear.py`) acts on the same rows: it adds half of a LightGBM's estimate of the residual left after KRS,
  computed from the line value, the K250 gap, the K250 spread, the airport and the UTC hour.

The pool weights, the standfloor / offblock_pull arms, the TW1′ shift and the KRS / KRS-NL coefficients are refitted from the 2025 fold vectors by `make constants` (`scripts/constants_2025.sh`; docs/REPRODUCE.md A3).

**Settings chosen by hand, not fitted.** These were set once, before scoring, and then checked by the gates of §7; none was tuned on the
leaderboard.
- Pool: one global weight per chain; never per airport.
- standfloor and offblock_pull: the candidate arms per airport come from small fixed grids (standfloor: window 10 or 15 min, base 0 or 300 s, step 0.5 or 1;
  offblock_pull: threshold 1.5 or 2 h, pull 0.25, 0.5 or 1, or a shift shrunk by `n/(n+50)`); the fit only picks one arm (or none) per airport.
  The fitted shifts are rounded to whole seconds.
- TW1′: the pair window of 300 s, "exactly two records, exactly one matched", and the shrinkage `n/(n+5)`; the shift is then rounded.
- KRS: three spread classes (terciles). KRS-NL: the shrink 0.5 on ĝ, the residual clip ±1,800 s and the LightGBM settings above.
- Unmatched and exposed rows (§6): the multipliers and clips of the Kumo in-context steps (F1 on linked other unmatched rows `1.0`,
  ±1,200 s; F3 on unlinked other unmatched rows `0.5`, ±600 s; F4 on unmatched LIRF rows `0.5`, ±1,200 s), the 95 % level of the bounds
  of PX-safe (the scenario-safe rule for exposed LIRF rows) and the "at least twice the bound" margin; the E1s cap is the largest value seen in 2025,
  not a tuned number.
- Schedule-copy rule: 3,600 s is the organisers' join edge as measured on 2025, and 8 h the scheduled-stamp window.

**All 2025 constants.** One file lists every fitted or hand-set constant of the final line with its provenance and the place in the
code that holds it: [`phase2/params/constants_2025.json`](../phase2/params/constants_2025.json). It documents the values; the scripts
keep their own copies, unchanged from the build of the submitted file, and `python tools/check_constants.py` checks that every value
listed there is the one in the code. Fitted files are written by `make constants` (`scripts/constants_2025.sh`) to the paths listed
under `fitted_files`, which `scripts/run_final.sh` checks before a run.

| constant | value | kind | provenance | held in |
|---|---|---|---|---|
| pool weights | dTK 0.3212, Kq 0.3065, NNA3 0.3723 weight | fitted on 2025 | global simplex fit on the 6-fold vectors of the three chains, matched 2025 rows (scripts/pool_weights_fit.py, make constants step pool); unrounded fit in phase2/params/pool_weights.json; the code uses the 4-decimal values (the 5-slot pool_mix call writes the NNA3 weight as 0.3722) | `phase2/params/pool_weights.json`; `phase2/run_pipeline.sh`; `phase2/steps/p2_v107.sh` |
| postrule arms | per airport, see file | fitted on 2025 | standfloor and offblock_pull: one arm (or none) per airport chosen from fixed grids on all 12 months of 2025 fold vectors (phase2/steps/p2_postrules.py fit, make constants step postrules); shifts rounded to 1 s | `phase2/params/postrules_fixed.json`; `phase2/periods/final.env` |
| postrule grids | standfloor window s 600/900, standfloor base s 0/300, standfloor step 0.5/1.0, offblock pull threshold s 5400/7200, offblock pull k 0.25/0.5/1.0, offblock pull shift shrink n/(n+50) | set by hand before scoring | candidate arms of the post-rule fit | `phase2/steps/p2_postrules.py` |
| tw1 shift | 1264 s | fitted on 2025 | TW1' LFPG twin rule: n/(n+5) * mean(y - pred) over the 2025 class rows, rounded (scripts/tw1_rule_build.py, make constants step tw1 -> tw1p_shift.json) | `phase2/steps/p2_v107.sh` |
| tw1 pair window | 300 s | set by hand before scoring | two departure records of one callsign at LFPG with \|dMVT\| <= 300 s form a pair | `scripts/tw1_rule_build.py` |
| tw1 shrink | 5 rows | set by hand before scoring | shrinkage n/(n+5) of the class mean | `scripts/tw1_rule_build.py` |
| krs beta | low spread 0.4520, mid spread 0.1726, high spread 0.0674 fraction of (K250 - line) | fitted on 2025 | per tercile of the K250 instance spread, fitted on the 2025 fold rows of the line (scripts/krs_fold_line.sh, make constants step krs) | `scratchpad/xchain2/v106K_K2_KRS_prod_coef.json` |
| krsnl shrink | 0.5 factor | set by hand before scoring | KRS-NL: line + 0.5 * g | `scripts/krs_nonlinear.py` |
| krsnl residual clip | 1800 s | set by hand before scoring | clip of the KRS-line residual the LightGBM is fitted to | `scripts/krs_nonlinear.py` |
| krsnl lightgbm | num leaves 63, min data in leaf 2000, rounds 300, learning rate 0.05 | set by hand before scoring | KRS-NL residual model settings; the fitted model is a 2025 fit (make constants step krsnl) | `scratchpad/xkrsnl/prod/gbm_A.txt`; `scripts/krs_nonlinear.py` |
| meta stacker clip | 150 s | set by hand before scoring | correction of the chain meta stacker clipped to +-150 s | `scripts/chain_stack.py` |
| clock stage clip | 300 s | set by hand before scoring | Huber clock stage (Hu6) correction clipped to +-300 s | `scripts/chain_clock.py` |
| callsign link step | k 1, clip s 1200 factor / s | set by hand before scoring | unmatched rows linked by callsign: + 1.0 * clip(g, +-1200) | `scripts/unm_link_lever.py` |
| kumo F1 F3 | F1 k 1, clip s 1200, F3 k 0.5, clip s 600 factor / s | set by hand before scoring | in-context residual on linked (F1) and unlinked (F3) unmatched rows | `scripts/kumo_icl_unmatched.py` |
| kumo F4 | k 0.5, clip s 1200 factor / s | set by hand before scoring | in-context residual on unmatched LIRF rows | `scripts/kumo_icl_lirf.py` |
| rule bag blend | lightgbm 0.5, catboost 0.5 weight | set by hand before scoring | seed-bag blend of the rule-priced rows (24 LightGBM + 16 CatBoost seeds) | `scripts/rulebag_blend.py` |
| pxsafe ord bounds | crossers 0.072, non crossers 0.259 share | measured on 2025 | 95 % exact one-sided upper bounds of the ORD share among 2025 exposed LIRF rows (0 of 40 crossers, 0 of 10 non-crossers; scripts/exposed_ord_bounds.py, make constants step exposed) | `phase2/steps/p2_exposed_scenarios.py`; `scripts/exposed_ord_bounds.py` |
| pxsafe margin | 2 factor | set by hand before scoring | a row moves only if its break-even ORD probability is at least twice its class bound | `phase2/steps/p2_exposed_scenarios.py` |
| e1s clock cap | 1822 s | measured on 2025 | largest at-stand clock among the 21 linked 2025 exposed non-LIRF rows (scripts/e1s_clock_cap.py, make constants step exposed) | `phase2/steps/p2_e1s.py`; `scripts/e1s_clock_cap.py` |
| tail guard threshold | 2400 s | chosen on 2025 from a fixed grid | KCATLg uses the raw-target member where its prediction is above 2,400 s; one global value from the grid {1,800, 2,400, 3,600 s, KCAT8 q99}, chosen on the 2025 folds | `scripts/kumo_kcat_logtarget_tailguard.py` |
| schedule copy join edge | 3600 s | measured on 2025 | the organisers' flight-plan join edge \|BLOCK - LOBT\| <= 3,600 s as measured on 2025 | `scripts/rule_schedule_copy_lirf_relink.py` |
| scheduled stamp window | 28800 s | set by hand before scoring | exposed rows: unmatched and MVT - SCHED > 8 h | `scripts/rule_schedule_copy_lirf_relink.py` |
| lobt bound | 3606 s | measured on 2025 | \|y - (MVT - LOBT)\| <= 3,606 s holds for every 2025 row with LOBT | `scripts/staged_delta_apply.py` |
| floor | 60 s | set by hand before scoring | prediction floor | `scripts/staged_delta_apply.py` |

## 6. Unmatched and exposed rows: label-free pricing
| class | steps |
|---|---|
| `unm_other` | C1 blend (V47L + CATV17x) → unm_resid residual stacker (LTFM kept at C1) → **callsign ADS-B link**: each row is linked to its own adsb.lol trace by callsign (IATA→ICAO map learned from matched rows, OpenFlights as fallback; take-off must match within 60 s) and a month-nested residual on the linked clock is applied (k 1.0, clip ±1,200 s) → Kumo in-context learner: **F1** on linked non-LTFM rows (16 clock inputs, `+1.0·clip(g, ±1,200)`), **F3** on unlinked non-LTFM rows (22 day-state and stand inputs, `+0.5·clip(g, ±600)`) |
| `unm_LIRF` | seed-bag blend FZ = 0.5·LightGBM (V25L s0–15 + V25Lb s10–17, 24 seeds) + 0.5·CatBoost (CATV9x s0–15, 16 seeds) → **F4**: Kumo in-context residual with the blend's own `p` and taxi as inputs, plus label-free stand and day inputs, `+0.5·clip(g, ±1,200)`; context = labelled 2025 `unm_LIRF` rows, leave-one-month-out → schedule-copy rule (below) |
| exposed | seed-bag blend FZ; PX-safe rule on LIRF rows (a row moves only if every label scenario, scheduled stamp, ordinary delay or day shift, agrees the move does not hurt; 2025 bounds); **E1s** on non-LIRF rows: `max(MVT − own at-stand ADS-B clock, 60)` when that clock is ≤ 1,822 s. Both thresholds come from fixed rules on 2025 rows: each PX-safe bound is the 95 % exact one-sided upper bound of the share of ORD labels among the 2025 exposed LIRF rows of its crossing class (0 of 40 crossers → 0.072, 0 of 10 non-crossers → 0.259), and a row moves only if its break-even ORD probability is at least twice its class bound (`scripts/exposed_ord_bounds.py`); the E1s cap is the largest at-stand clock among the 21 linked 2025 exposed non-LIRF rows (`scripts/e1s_clock_cap.py`) |

The seed bag is ordinary seed averaging: it cuts the blend's seed-to-seed SD on these rows from ≈ 320 s (15 + 2 seeds) to ≈ 126 s.
The Kumo in-context learner here is Kumo Tabular (medium, 4 estimators, CPU).

**Schedule-copy rule at LIRF (join-rule regime call).** The organisers' flight-plan join pairs a block time with an NM
flight only if `|BLOCK − LOBT| ≤ 3,600 s`; measured on 2025, this edge is hard. When an unmatched LIRF departure's NM record can still be
recovered (the orphan arrival record of the same flight at its destination: same route, landing 10 min–8 h after take-off, scored on
flight-number digits, aircraft type and schedule, unique best match only), the join failed for one of two reasons. With
`L = MVT − LOBT_rec`, `A = MVT − AOBT_3,rec`, `MS = MVT − SCHED`, on unmatched LIRF rows with `MS ≤ 8 h`:

| cell | condition | meaning | prediction | 2025 rows | share with BLOCK = SCHED |
|---|---|---|---|---|---|
| ORD | `|A − L| > 3,600` | the real off-block was > 1 h from LOBT; the stamp is real | `max(A + off_med, 60)` | 100 | 0.00 |
| SUB | `|A − L| ≤ 3,600` and `|MS − L| > 3,600` | a real off-block would have joined, so BLOCK is the substitute scheduled time | `max(MS, 60)` | 28 | 0.93 |
| other | neither | | unchanged | 9 | |

`off_med` is the airport median of `y − (MVT − AOBT_3)` on matched 2025 rows; 3,600 s is the measured join edge, not a fitted value. The
rule runs last (`scripts/rule_schedule_copy_lirf_relink.py`, `scripts/rule_schedule_copy_lirf_apply.py`). On the final file it changes
69 rows; every cell margin is ≥ 60 s, so the ±12 s re-jitter of the final file cannot flip a cell. 2025 check, leave-one-month-out:
ΔMSE −429.7 s² (units of §7; 11 of 12 held-out months better); a placebo that permutes the cells within month costs +1,281.8. The rule was derived on
the same 2025 rows, so this check is a confirmation, not fresh evidence; nothing besides `off_med` is fitted.

## 7. Validation
Fitting uses the 2025 labels, label-free statistics over the ranking rows (§9 items 1 and 2), the published daily PRU delay aggregates
of the scored days ((B), `xpr`) and the pseudo-labels of the run's own first pass (§9 item 3); no 2026 label exists in the pipeline. Every leaderboard upload was a complete model built by these scripts from 2025 fits, and no weight, clip,
threshold or other continuous parameter of the submitted line was set from a leaderboard score.

**Use of the public leaderboard.** The public leaderboard (January and July 2026) was used as a confirmation check of changes that had
passed the pre-declared 2025 validation gates of this section (the month-pair folds, the fresh folds and the sealed days). Five
components of the submitted line did not pass every criterion of their gate: E1s, the wide-leaf twin, the pseudo-label member, KRS-NL
and KCAT8 (the last later replaced by KCATLg, which passed). They were uploaded as isolated changes, and the choice to keep them or roll
them back was taken on the public score; so were the parts of the combinations of uploads 69, 85 and 91 (91, on 1 October, was chosen over 90) and, before the fixed
gate, dropping the day-of-year feature against the 2025 folds (14 September). These were discrete choices
between pre-built files. Every upload, the 2025 check that preceded it and its public score are listed in [`upload_log.md`](upload_log.md).
The effects reported in this document are 2025 validation results.

**Folds.** Six month-pair folds: 3+9, 2+8, 4+10, 1+7 (gate folds, each a winter and a summer month like the test) and 5+11, 6+12 (fresh
folds, added on 25 September; not used to choose between candidates of the matched-row line and decisive for any re-weighting of the
chain; for four unmatched-row components they served to choose between arms, see the table below). 20 % of airport-days (720 days) are
sealed: drawn on 29 September with a fixed seed before any score on them (`scripts/sealed_days.py`), read once per candidate;
components admitted before that date were not checked on them. Every stage is fitted nested, so its
own fit leaves the scored fold out. Fold errors are re-weighted to the ranking-set mix of airport × link state × row class × anchor
source × season (label-free keys only). Results are given as ΔMSE in s² on these re-weighted fold rows (a negative value is a gain;
at the 2025 matched-row RMSE of 166.5 s of the final line, 1 s of RMSE ≈ 333 s² of MSE).

**Gate for a matched-row change.** Gain on ≥ 3/4 gate folds (4/4 for late levers) and also without the 3 largest row contributions;
4-fold mean below a bar that grows with the number of arms tried; the gain survives removing each airport-day's mean error; both fresh
folds better; sealed days confirm; better than a placebo (inputs or context labels permuted within airport × day); and a label-free
2026 drift check of every new input. Small classes (unmatched rows) use leave-one-month-out instead (≥ 7/12 held-out months better).

**2025 gate results of the main components.** Each row is the gate that admitted the component, in the order of admission. Metric:
ΔMSE in s² on the scored 2025 fold rows, the line with the component minus the line without it *at the time of admission*, so the
reference line differs from row to row and the rows do not add up to the total. Weighting: **w** = fold rows re-weighted to the
ranking-set mix as described above (an earlier form of it before 25 September); **r** = plain row mean (the first protocol). Gate folds 3+9 / 2+8 / 4+10 / 1+7; until
20 September the protocol had three folds (3+9 / 2+8 / 4+10) with 1+7 as confirmation. Fresh folds 5+11 / 6+12 exist from 25 September
and the sealed days from 29 September; "—" means the check did not exist yet. Unmatched-row components are scored on their own class
with leave-one-month-out (LOMO) fits; the number of held-out months that improve is given in brackets. For the unmatched-row components of 28 September to 2 October,
the folds 5+11 and 6+12 were used to choose between arms of the same component, so they are marked "selection" and are not fresh there. The time-ordered list
of uploads with these checks is [`upload_log.md`](upload_log.md).

| component | gate folds 3+9 / 2+8 / 4+10 / 1+7 | fresh 5+11 / 6+12 | sealed days | wt | verdict |
|---|---|---|---|---|---|
| mixture members, residual target to `MVT − AOBT_3` | before the protocol: holdout RMSE 402 → 366 s | — | — | r | pre-protocol |
| neural member (RealMLP) | 3 folds mean −348.5 (3/3); 1+7 −422.3 | — | — | r | pass |
| meta stacker | −141 / −596 / −241; 1+7 −409 | — | — | r | pass |
| ADS-B ground clocks in the members | −1,072 / −3,634 / −4,402; 1+7 −4,718 | — | — | r | pass |
| stand-position clocks in the members | −5,610 / −2,088 / −1,609 | — | — | r | pass |
| fallback probability `q` as stacker input | −111 / −247 / −51 / −56 | — | — | r | pass (1+7 bootstrap interval crossed 0) |
| runway-queue members | −140 / −70 / −172 / −244 | — | — | r | pass |
| ADS-B link flag as stacker input | −79 / −74 / −75 / −107 | — | — | w | pass |
| C4 | −344 / −382 / −350 / −516 | — | — | w | pass |
| tree D (over C4) | −209 / −93 / −137 / −247 | — | — | w | pass |
| ADS-B clock-residual stage | −602 / −594 / −301 / −801 | — | — | w | pass |
| Huber clock stage, all stages fitted on 12 months | −181 / −178 / −291 / −150 | — | — | w | pass |
| leave-row stage (A0 + L) | −32 / −43 / −42 / −57 | better on both | — | w | pass |
| three-chain pool | −192 / −266 / −240 / −142 | not recorded | — | w | pass |
| chain bundle: airport-dedicated and clock-free members, jittered clock stage | −128 / −149 / −101 / −147 | not recorded | — | w | pass |
| push-back clocks, larger Huber stage | −34 / −32 / −50 / −24 | not recorded | — | w | pass |
| (B) daily calibration | −207 / −629 / −290 / −346 (12/12 months) | not recorded | — | w | pass |
| PRU daily indicators as stacker inputs | −270 / −434 / −382 / −333 | not recorded | — | w | pass |
| standfloor | −42 / −157 / −42 / −88 (placebo 0) | not recorded | — | w | pass |
| offblock_pull | −49 / −7 / −49 / −52 (placebo +0.1) | not recorded | — | w | pass |
| top-10 raw features as stacker inputs; LIRF out of the first clock stage | −70 / −85 / −55 / −51 | not recorded | — | w | pass |
| push-start clock in the Huber stage | −60 / −48 / −5.5 / −48 | not recorded | — | w | pass |
| longest-stop clocks in the Huber stage | −29.6 / −7.4 / −16.1 / −36.9 | not recorded | — | w | pass |
| wide-leaf twin (chain isolate) | −19.8 / −1.1 / −20.3 / −13.5 | not recorded | −2.6 (2/4 < 0) | w | main rule pass; day-adjusted rule failed on 1+7; kept on the public score |
| pseudo-label member (chain isolate) | −16.0 / −76.0 / −24.8 / **+4.4** | not recorded | not recorded | w | failed the main rule on 1+7; kept on the public score |
| unmatched: callsign ADS-B link, half strength | −114.0 / −184.9 / −74.9 / −196.5 (12/12 months; placebo +0.8) | −118.6 / −138.6 (selection) | — | w | pass |
| unmatched: the same at full strength (k 1.0, ±1,200 s) | −61.0 / −98.5 / −36.8 / −131.8 vs half strength (12/12) | −57.8 / −66.0 (selection) | — | w | pass |
| exposed: E1s (11 rows) | no fold gate; own clock better than the proxy on 13 of 21 rows (75 % declared) | — | — | — | failed its criterion; kept on the public score |
| exposed: PX-safe (8 rows) | no fold gate: a row moves only if every label scenario agrees | — | — | — | rule |
| unmatched: F1 (linked rows) | −36.9 / −44.1 / −35.1 / −138.9 (11/12 months; placebo +241.8) | −138.0 / −11.3 (selection) | — | w | pass |
| unmatched: F3 (unlinked rows) | −16.0 / −21.5 / −21.9 / −39.0 (10/12 months; placebo −17.2) | −3.9 / −0.5 (selection) | — | w | pass (narrow margin over its placebo) |
| unmatched LIRF: F4 | LOMO −60.1 / −58.6 / −63.1 / −7.6 (12/12 months; placebo −4.5) | — | — | w | pass |
| NBR_CLOCK | −40.7 / −44.9 / −30.1 / −14.3 | member level −95.7 / −57.7 | not recorded | w | pass |
| K250 | −451.2 / −551.9 / −405.1 / −511.3 | −475.2 / −429.8 | not recorded | w | pass |
| KRS | −2.9 / −159.0 / −42.0 / −166.1 | −91.9 / −125.9 | 3/4 folds < 0 | w | pass (the fold values had been seen in an exploratory step; the sealed days decided) |
| K250 in the clock stages (KSTK_CLOCK) | −106.0 / −131.2 / −103.8 / −101.7 | −101.6 / −120.8 | not recorded | w | pass |
| airport-day set transformer, first form | −47.2 / −73.7 / −22.9 / −70.8 | −20.7 / −50.2 | −49.4 (3/4 < 0) | w | pass |
| BIG (replacing the first form) | −8.5 / −30.5 / −57.3 / −209.9 | −134.6 / −82.5 | −81.3 (4/4 < 0) | w | pass |
| TW1′ (LFPG twin rule) | LOMO −163.7 / −49.5 / −152.3 / −97.8 (11/12 months; placebo 0.03) | −85.8 / −141.7 | not recorded | w | pass |
| KSTK (line with and without) | −67.4 / −27.2 / −41.3 / −41.6 | −27.7 / −16.3 | −103.4 (4/4 < 0, chain level) | w | pass |
| KRS-NL | −35.4 / −102.2 / **+3.8** / −48.0 | −25.1 / −17.3 | −83.3 (3/4 < 0) | w | failed the 4/4 rule on 4+10; kept on the public score |
| KCAT8 in the Kumo-member variant step | −100.5 / −95.6 / −93.3 / −210.1 | −87.6 / −91.9 | −97.8 (4/4 < 0) | w | failed the month-cell rule (EHAM April +286.5); documented exception, replaced by KCATLg |
| schedule-copy rule at LIRF | LOMO −429.7 (11/12 months; placebo +1,281.8) | — | not recorded | w | pass (rule derived on the same rows; see §6) |
| KCATLg (vs KCAT8 in that step) | −34.7 / −33.0 / −41.6 / −68.7 | −56.3 / −16.2 | −57.9 (4/4 < 0) | w | pass |

**Validation of the final line by month and airport.** Matched 2025 rows, each month scored by a model that did not see it (chain line
with KRS; RMSE in s):

| month | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | 167.2 | 172.2 | 145.9 | 154.4 | 157.1 | 182.9 | 200.4 | 177.3 | 167.3 | 163.0 | 145.4 | 151.7 |
| mean error | −1.5 | +0.1 | −2.0 | −7.8 | −1.9 | −2.8 | −1.4 | −0.4 | −0.6 | +0.1 | +1.0 | +0.2 |

| airport | Jan | Feb | Jun | Jul | | airport | Jan | Feb | Jun | Jul |
|---|---|---|---|---|---|---|---|---|---|---|
| EDDF | 117 | 81 | 124 | 103 | | LEMD | 148 | 139 | 135 | 154 |
| EDDM | 83 | 61 | 118 | 121 | | LFPG | 235 | 183 | 213 | 216 |
| EGLL | 176 | 146 | 209 | 218 | | LIRF | 265 | 274 | 363 | 393 |
| EHAM | 118 | 74 | 110 | 95 | | LSZH | 98 | 57 | 84 | 80 |
| LEBL | 91 | 79 | 141 | 219 | | LTFM | 194 | 321 | 188 | 230 |

The full 10 airports × 12 months are shown in [`figures/airport_month_rmse.png`](figures/airport_month_rmse.png). The chain-variant steps
admitted after KRS (K250 in the clock stages, BIG, KSTK, each with its KRS refit) lower the plain row-mean MSE in all 12 held-out
months: −164 s² on the 12-month mean, −251 s² on January + July, −161 s² on the eight months February–June and August–October; each
single step improves 10 to 12 of the 12 months. The errors track the share of early-block rows (block more than
10 min before `AOBT_3`): 30 % of the squared error in March, 51 % in July. At LTFM in February the snow days 19 and 23 Feb 2025 hold 6 % of the rows and 46 % of the squared error
above the other months' level (`model_card.md`); LIRF summer is the dark hub
with scheduled-stamp fallbacks. The last two levers on top of this line, measured on the same folds against the line without them:

| lever | gate folds 3+9 / 2+8 / 4+10 / 1+7 (ΔMSE, s²) | fresh 5+11 / 6+12 | sealed days |
|---|---|---|---|
| KRS-NL | −45.5 (4-fold mean) | −25.1 / −17.3 | −83.3 |
| KCATLg in the Kumo-member variant step (vs KCAT8 there) | −34.7 / −33.0 / −41.6 / −68.7 | −56.3 / −16.2 | −57.9 |

KCATLg also improves every airport and every month on the 12-month pooled line (−193 s² of ΔMSE); its gain grows with the prediction
size (top decile −167, top 1 % −333 s² of ΔMSE) and the mean signed residual of the top 1 % is unchanged, so the guard leaves no tail
under-prediction.

## 8. Final assembly
For a ranking file (`scripts/run_final.sh <period>`):
1. Workspace (`phase2/p2ws.py`): a copy of `scripts/`, link farms over the 2025 data, a copy-on-write guard and a ledger of every file read.
2. Period inputs: features and side tables for 2025 + the ranking rows; ADS-B clocks of the ranking days; PRU, OPDI, METAR coverage check.
3. Members: every seed of `scripts/xsave/members_xsave.tsv` fitted on all 2025 rows; NBR_CLOCK MBm / MA; BIG (3 seeds).
4. Kumo: K250, KCAT8 and KCATL instances on the ranking rows, the KCATL16 bag and the tail guard (`scripts/kumo_prod.sh`).
5. Base line (`phase2/run_pipeline.sh`): the three chains, the wide-leaf and pseudo-label isolates (with the pseudo-label member retrained in the run), pool, (B), unmatched and exposed
   rows by rule (§6), standfloor, offblock_pull; then KSTK on the chain-stacker frames.
6. Chain-variant steps 1 to 5 (§4), TW1′, KRS and KRS-NL (`phase2/steps/p2_v107.sh`) → the line output `submissions/final.v111.parquet`.
7. The schedule-copy rule at LIRF, last → `zesty-puzzle_final.parquet`, checked by `scripts/validate_submission.py` (ids, schema,
   ≥ 60 s, `|pred − (MVT − LOBT)| ≤ 3,606 s`).

**The submitted file.** It is the two-line assembly `scripts/assemble_final.sh`: line A (the KCATLg line) and line B (the same line
with the 24 + 16 seed bag), plus the schedule-copy rule at LIRF. The original build combined four saved runs whose K250 and KCAT8 deltas
cancel; the account is in [`REPRODUCE.md` §6b](REPRODUCE.md#6b-the-staged-assembly-of-the-build).

The file equals line A on all 661,733 matched and 8,395 other unmatched rows. It equals line B on the 70 exposed and 523 unmatched-LIRF rows, and the schedule-copy rule on its 69 rows. All unmatched and exposed rows of the four
months are priced by the rules of §6.
Public leaderboard score of its January and July rows on the public ids: 221.4616.

## 9. Treatment of public and hidden months
The final file covers January, February, June and July 2026. January and July are the months of the public ranking file; February and
June were never on a leaderboard. **All four months go through the same code path:** both line runs of the submitted file (§8), and the single-pass recipe
`scripts/run_final.sh`, reads the final ranking file `final_ranking.parquet`, which holds
all four months, and applies each step to all of its rows in one pass. No step of these runs selects ranking rows by month or reads an earlier submission,
and no departure label of 2026 exists anywhere in the pipeline: the block and taxi-out times of ranking departures are hidden in the
competition files.

Rows of the public ranking file (January and July 2026) enter the computation in exactly three places. None of them reads a label:
1. **Pooled feature statistics of the 2025 side** ([`REPRODUCE.md` §5](REPRODUCE.md#5-a-2025-side-once)). Part A, the fold vectors and the
   saved member fits, was built with the public ranking file in `data/raw/`. A few statistics of `scripts/build_features.py` are pooled
   over the training and ranking rows; they read only columns that the ranking file provides (schedule, movement and flight-plan times,
   and the records of arrivals, whose block times are given).
2. **The arrival statistics of the `arrfb` group** ([`known_issues.md` §6](known_issues.md#6-lgb_adsb_linked_a-refit-sensitivity-to-the-feature-generation-version-at-ltfm)).
   The per airport × operator share of arrivals whose block time equals the scheduled time (`arrfb_op`) is a mean over all arrival
   records of the training and ranking files. The saved member fits were made on the feature tables of the public period, so the public
   January and July arrivals are part of this mean on the 2025 training rows. Rebuilding the tables with the final ranking file moves
   these values by about 10⁻⁴.
3. **The pseudo-label member** (`pseudolabel_pl3p88`, §3). It is retrained in every production run on the 2025 rows plus the matched
   ranking rows of the file being predicted. The ranking rows carry pseudo-labels: the first-pass prediction of the same run, at weight
   0.3, kept only where the member's fallback probability is below 0.05 and the first-pass prediction below 7,200 s. In the final run
   these are the matched rows of all four months of the final ranking file.

Why this is label-free: the only 2026 targets the pipeline ever uses are its own predictions (item 3); items 1 and 2 use columns that
the organisers publish for every ranking row. The daily calibration (B) and the `xpr` stacker inputs use no taxi-time label of the
ranking rows; they use EUROCONTROL's published daily PRU delay aggregates of the scored days (`DATA_SOURCES.md` row 2 and the
24–25 Sep ruling of its §4). **February and June rows are treated identically to January and July rows.** In the final
run every statistic over ranking rows, every transductive feature (same-day neighbours, day-level delay regime, flight memory), the
pseudo-label mask and every rule of §6 is computed in the same way for all four months. The only asymmetry sits on the
2025 side (items 1 and 2): the training-row statistics were pooled with the public January and July arrivals rather than with all four
months. This changes no rule and no selection, and it changes the affected training-row values by about 10⁻⁴.

### Public-month probes
Uploads 13, 23, 52 and 107 changed rows of one public month only; upload 8 is listed because its log entry names January + July. Scores
and row counts are those of [`upload_log.md`](upload_log.md).

| upload | what it was | rows changed | in the submitted file |
|---|---|---|---|
| 8 | LightGBM members trained for a fixed 1,000 trees instead of early stopping on the **2025** January + July fold (labelled 2025 months; no 2026 row was used for stopping) | all 344,841 public rows (new member fit) | the fixed tree count was kept: every LightGBM and CatBoost member of `scripts/xsave/members_xsave.tsv` runs with `--fixed-rounds`, so its tree count does not depend on any fold score |
| 13 | the day-shift value (prediction = 86,400 s + 961 s) applied outright to the July 2026 LIRF unmatched rows with `MVT − SCHED` > 16 h; a test of one mechanism | 4 | no; rolled back (280.448 vs 277.647 for its base, upload 12). Exposed rows of the final file are priced by the seed-bag blend, PX-safe and E1s (§6) |
| 23 | upload 22 with its January 2026 rows taken from upload 19; a diagnostic of whether a loss sat in January or July | 152,719 (all January rows) | no; diagnostic only (277.005 vs 276.729 for upload 22). The final file takes every month from the same line runs (§8) |
| 52 | January 2026 level shrink on matched rows at LTFM, EDDM, LSZH and EDDF: `pred − 0.5·E·(1 − p)` with E = +136 / +82 / +93 / +79 s, weight 0.5 fixed before the upload; a single test, no follow-up | 58,048 | no; rolled back (242.0357 vs 241.9536 for its base, upload 50). No step of the final path selects ranking rows by month |
| 107 | −22 s on the matched LIRF July 2026 rows, from the published PRU July 2026 delay level (+33.0 s) minus the 2025 out-of-fold level (+11.2 s), shift fixed before the upload | 15,557 | no; rolled back (223.7937 vs 223.7028 for its base, upload 104). No step of the final path selects ranking rows by month |

Month expressions in `scripts/` and `phase2/` (`dt.month()`, `is_in([1, 7])`) are of four kinds, and none gives ranking rows of one
month a different treatment in the submitted file:
- printouts that write no value: the January + July 2026 coverage of `scripts/pru_build.py`, the Jan / Jul check table of
  `scripts/build_dreg_features.py`, the Jan / Jul cross-check of `scripts/rule_schedule_copy_lirf_relink.py` against the public-period run;
- selections of 2025 training rows: the dry run of `phase2/steps/p2_mkdry.py` (`is_rank == 0`), the `sim` mode of `scripts/newstand_map.py`,
  the leave-one-month-out gate of `scripts/tw1_rule_build.py`;
- month as a key computed in the same way for every month: the printed per-month means of `combo_c4_apply.py` and `isolate_delta.py`,
  the calendar-month column of `pru_daily_calib.py` and of the 2025 fold frame of `p2_postrules.py`, the summer-time offset of the
  trainers and `p2_unmatched.py`;
- the public-period check file of step F4 (`scripts/public_period_extract.py`), which selects the public ids and is not the submitted file.

Short labels used in code and tables are listed in [`naming.md`](naming.md).
