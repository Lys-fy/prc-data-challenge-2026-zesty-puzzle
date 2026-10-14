# Baseline

A laptop-scale version of the core idea of the full model (`docs/METHOD.md`): 298 lines of Python (`baseline/*.py`), one LightGBM model, competition files only, no external data.

## What it does
1. **Anchor.** Predict the residual `y − (MVT − AOBT_3)` (METHOD §1). Rows without an NM record have no `AOBT_3`; their anchor is the airport median taxi-out of the unmatched training rows.
2. **One LightGBM model** (600 rounds, 127 leaves, learning rate 0.05, seed 0, deterministic; `config.py`) on 21 features:
   - categorical: airport, runway, stand, aircraft type, operator (NM operator, else the letters of the flight number), wake category, flight type;
   - calendar: UTC hour, weekday;
   - NM deltas: `MVT −` `AOBT_3`, `LOBT`, `EOBT_1`, `IOBT`, `SCHED`; `AOBT_3 − LOBT`; matched flag;
   - congestion counted from the rows of the same file: departures 15 min before and after take-off, arrivals within ±15 min, aircraft taxiing out (`AOBT_3` passed, not yet airborne) at take-off and at the departure's own `AOBT_3`.
   Output: anchor + residual, kept within `MVT − LOBT ± 3,606 s` where `LOBT` exists, floored at 60 s (METHOD §3).
3. **Schedule-copy rule** for unmatched LIRF rows (METHOD §6), applied last. `rule.py` re-implements the orphan-arrival link of `scripts/rule_schedule_copy_lirf_relink.py` (function `match`) and the ORD / SUB cells; `off_med` comes from the training rows. On 2025 it links the same 175 LIRF rows as the full-pipeline script; on the final ranking file it changes 69 rows, as in the full pipeline.
4. **Evaluation** on the 2025 month-pair folds of METHOD §7: fit on ten months, score the two held-out months. Plain row RMSE (no re-weighting to the ranking-set mix).
5. **Prediction**: fit on all of 2025, predict the ranking file, write `MVT_ID_mvt` (Float64) and `TAXITIME_SEC_mvt` (Int32), the schema of `final_submitting.parquet`.

## Run
```bash
make data                                   # competition files -> data/raw/, data/final/ (team credentials)
make baseline                               # = the two commands below; outputs in data/baseline/ (git-ignored)
python -m baseline.run evaluate --raw-dir data/raw --out data/baseline [--folds 1+7,2+8,3+9,4+10,5+11,6+12] [--threads 8]
python -m baseline.run predict  --raw-dir data/raw --ranking data/final/final_ranking.parquet --out data/baseline
python -m pytest baseline/tests             # synthetic rows: anchor, rule cells, link + apply
```
Needs polars, numpy and LightGBM (the main environment, `uv sync`). `evaluate` writes `folds.json`; `predict` writes `baseline_<ranking stem>.parquet`.

## Results (2025 folds, measured 13 Oct 2026)
CPU, 8 threads, polars 1.44.1, LightGBM 4.7.0. RMSE in s. "anchor only" is `MVT − AOBT_3` without the model.

| fold | rows | all | matched | matched, anchor only | unmatched | unmatched without the rule | rule rows |
|---|---|---|---|---|---|---|---|
| 1+7 | 344,419 | 437.3 | 244.3 | 428.2 | 2,915.9 | 2,958.7 | 50 |
| 2+8 | 334,914 | 349.5 | 233.8 | 382.3 | 2,577.5 | 2,583.4 | 22 |
| 3+9 | 348,399 | 302.6 | 215.2 | 377.6 | 2,122.1 | 2,129.5 | 17 |
| 4+10 | 360,962 | 249.1 | 214.5 | 369.4 | 1,405.5 | 1,429.5 | 19 |
| 5+11 | 347,534 | 314.3 | 209.1 | 366.4 | 2,554.4 | 2,564.2 | 20 |
| 6+12 | 348,819 | 335.0 | 222.5 | 383.2 | 2,309.6 | 2,400.5 | 33 |

For comparison, the full pipeline's 2025 out-of-fold RMSE on matched rows is 166.5 s (METHOD §7).
Unmatched rows are 0.9–1.6 % of a fold; their RMSE is dominated by the 113 exposed 2025 rows (`MVT − SCHED > 8 h`, label SD 30,797 s).

Runtime (wall clock): `evaluate` with all six folds 205 s (about 33 s per fold, peak memory 7.0 GB); `predict` on the final ranking file
(670,790 departures) 46 s (6.6 GB). Two runs gave bit-identical submissions.
