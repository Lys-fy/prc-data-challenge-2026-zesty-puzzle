# Model card: zesty-puzzle taxi-out model

## Model details
- **Task.** Regression: taxi-out time (`MVT_TIME − BLOCK_TIME`, seconds) of a departure, as recorded by the departure airport.
- **Airports.** EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.
- **Type.** A stacked ensemble: mixture members (LightGBM, CatBoost, RealMLP), a tabular foundation model used in context (NVIDIA Kumo
  Tabular, open weights, run locally), an airport-day set transformer, three LightGBM stacking chains with correction stages, label-free
  post-rules, and separate paths for flights without a flight-plan match (`docs/METHOD.md`).
- **Version.** The submitted file is reproduced by the two-line assembly `scripts/assemble_final.sh` (line A = KCATLg line, line B = bag line); the original four-run staged build gives the same bytes (code at git tag `submission-build`, `docs/REPRODUCE.md` §6b). `scripts/run_final.sh` runs every stage of both lines. Author Luis Melzer, team zesty-puzzle, PRC Data Challenge 2026. Licence GPL-3.0.

## Intended use
- Research and benchmarking of taxi-out prediction **after the fact**, on the challenge's data format. The model uses information that is
  only available once the flight has left: the Network Manager's actual off-block time (`AOBT_3`), the movement time `MVT_TIME` itself
  (the target is defined relative to it), ADS-B ground traces of the departure, and same-day neighbours (earlier and later departures).
- **Not intended** for operational, real-time or safety-related use: it is not a predictor at the time of off-block and has not been
  validated for that.

## Training data
- 2,085,047 departures of 2025 at the ten airports (competition data; EUROCONTROL Network Manager flight-plan fields and airport movement
  records). Not redistributed.
- Open data: adsb.lol ADS-B traces (ODbL 1.0), EUROCONTROL PRU daily indicators and OPDI milestones, IEM METAR, OpenFlights airline
  designators (`docs/DATA_SOURCES.md`).
- The labels contain recording artefacts (block time = scheduled time, block time on the wrong date). They are part of the test labels as
  well (organiser statement, 9 Sep 2026) and are modelled, not removed.

## Evaluation
| set | departures | RMSE (s) |
|---|---|---|
| public leaderboard, Jan + Jul 2026 (Jan/Jul rows of the final file) | 344,841 | 221.46 |
| final blind ranking, Jan + Feb + Jun + Jul 2026 | 670,790 | published by the organisers after the challenge |
| 2025 month-held-out folds, matched rows, final line (v111) | 2,062,623 | 166.5 (per month and airport, chain line with KRS before KRS-NL, overall 166.6: `docs/METHOD.md` §7) |
The error is dominated by a small tail: labels over 2 h are 0.03 % of rows and 46 % of the squared deviation; matched departures without an adsb.lol
trace (920,353, 44.6 % of matched rows) carry 70 % of the matched squared error (`docs/figures/row_types_2025.json`).

## Limitations and risks
- **Coverage-dependent quality.** Matched 2025 rows without an adsb.lol trace have 208 s out-of-fold RMSE against 124 s with one. In the
  final months the linked share is 0.00 at LTFM, 0.14 at LFPG and 0.30 at LIRF (0.62–0.96 elsewhere).
- **LTFM has no ADS-B ground clock.** Its 2025 out-of-fold month RMSE (matched rows, `docs/figures/airport_month_rmse_2025.json`) is
  184–202 s in 10 of 12 months, 230 s in July and 321 s in February. The two worst February days are the snow days 19 and 23 Feb 2025
  (day RMSE 698 and 788 s, each with 17 taxi-outs of 1.5–2.1 h); they hold 6 % of the February rows and 46 % of its squared error above the
  other months' level, and February without them is 274 s.
- **Recording artefacts dominate the tail.** For unmatched LIRF rows the model predicts a probability-weighted mix of "real taxi" and
  "fallback stamp"; individual predictions there can be far off in either direction.
- **Distribution shift.** Every member is fitted on 2025 labels; 2026 enters only through label-free ranking-row statistics, the
  published daily PRU delay aggregates of the scored days ((B), `xpr`; no taxi-time label) and the pseudo-label member's first-pass targets (`docs/METHOD.md` §9). A new period with other airports, other recording practices or missing open data
  (OPDI, PRU) degrades the stages that depend on them; `docs/REPRODUCE.md` lists the fallbacks.
- **Not bit-reproducible.** Multi-threaded float sums in two polars group-bys of the fold side move 1,106 rows on two days by up to ±3 s between runs; separately, LightGBM builds different trees on aarch64 and x86_64 and a refit with the same seed is a new draw (`docs/known_issues.md`).
- **Transductive features.** Predictions for one departure depend on the other departures of the same day in the ranking file
  (allowed by the organisers, 9 Sep 2026). Predictions for a single flight in isolation are not supported.

## Ethical and legal notes
- No personal data are used. Airframe identifiers in ADS-B traces are used only to link a departure to its own trace; no table of them is
  published. Operator identities are anonymised in the competition data and were not de-anonymised.
- No proprietary model or paid API is used. The OpenSky Network historical database is not used.
