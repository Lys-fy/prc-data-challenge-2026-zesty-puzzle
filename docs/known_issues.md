# Known issues

## 1. The build is not byte-exact
- **polars group-by float sums.** Two fold-side meta inputs are computed with multi-threaded polars group-bys. The order of the float sums
  changes between runs, which moves 1,106 rows on two dark days (LIRF 2026-07-15, LTFM 2026-01-31) by up to ±3 s (≈ 1e-4 s RMSE).
- **LightGBM refits are new draws.** The same command, seed and inputs give a different model on another run and, further, on another CPU
  architecture (aarch64 vs x86_64). Seed-to-seed spreads: 24–33 s RMS for `lgb_adsb_linked_a`; a same-machine refit of one seed moves 3–6 s RMS.
  Chain variants are therefore always measured against a twin built on the same machine.
- **Kumo Tabular in fp16 on a GPU** moves 0.4–1.3 s RMS per instance between GPU types (correlation ≥ 0.99995). Repeated runs on
  the same GB10 GPU were bit-identical (184 of 184 re-run instances).
- **CatBoost on GPU** is not bit-identical between GPU types (no size was measured); a saved CatBoost model re-predicts bit-exactly.

## 2. Inputs that change between issues of a ranking file
The final ranking file was re-issued by the organisers on 8 Oct 2026 with every movement time re-jittered by up to ±12 s and `BLOCK_TIME`
changed on 617k rows. Everything that depends on the ranking rows (features, side tables, ADS-B clock extraction, member predictions,
Kumo) must be rebuilt from the new file; a saved member moves 9–11 s RMS between the two issues (LIRF 18–26 s). The schedule-copy rule at
LIRF has a margin of ≥ 60 s in every cell, so the re-jitter cannot flip a cell.

## 3. Coverage of the open data
- adsb.lol: per airport, the share of departures linked to an ADS-B trace in the final months is LFPG 0.14, LIRF 0.30, LTFM 0.00 and
  0.62–0.96 elsewhere; days missing from the main release are pulled from the other release tags. Rows without a link use the
  clock-free branches.
- PRU daily indicators: days without PRU data get a zero calibration shift in (B).
- OPDI: `scripts/fetch_opdi.py --periods` must include the months of the ranking file.

## 4. Leakage audit: the LTFM row-identifier order (not used)
At LTFM, in October and November 2025, the row identifiers of the competition export follow the hidden off-block order within a take-off minute
(97.5 % of 7,239 adjacent pairs in October and November 2025). Measured on 2025 (train on October, test on November and the reverse), a
feature built from that order would lower the LTFM departure RMSE by about 4.5 s. No feature, rule or selector reads identifier values or
identifier order; identifiers are used only as join keys, and the TW1′ rule explicitly ignores record order.

## 5. Leakage audit: target encodings in the 2025 folds
The out-of-fold target encodings are computed leaving the scored fold out; an audit of encodings that pair a month with its fold partner
found no material effect on the fold scores (< 1 s² of ΔMSE per fold).

## 6. `lgb_adsb_linked_a` refit sensitivity to the feature-generation version at LTFM
Refitting `lgb_adsb_linked_a` (the clock-anchored member, ADS-B-linked rows) from scratch on the feature tables of the final ranking file
moves its LTFM level by −17 to −24 s (airport × month mean); its seed-to-seed spread at LTFM is about 1 s, and the general refit spread
of issue 1 comes on top. The saved seeds s0–s2 were fitted on feature tables whose pooled `arrfb`, target-encoding and queue statistics
included the public ranking file; the tables of the final ranking file change these pooled values at LTFM. The 89,793 LTFM ranking rows
are affected. The submitted file uses the saved s0–s2 predictions, which re-predict bit-exactly from the saved models, so its impact is
0. A from-scratch refit of this member therefore gives an LTFM level 17–24 s below the saved seeds; the cause is the changed pooled
features, not a code fault.

## 7. What is byte-verified and what is not
- **Byte-verified from saved stage inputs** ([`REPRODUCE.md` §6c](REPRODUCE.md#6c-stage-by-stage-md5-chain)): with the code of this
  repository, from the saved chain outputs, fold vectors, Kumo in-context arm files and stage slots of the two line workspaces, every
  intermediate of both lines (`v99_iso_*`, compose, E1s, F1/F3, F4 → `final.v99`; the chain-variant isolates, v100 → v106K, KRS → `final.v111`)
  and the assembly (`F2_merge` → `zesty-puzzle_final.parquet` `1a155e9f…` → `public_janjul.parquet` `9cd6296d…`) reproduce the saved md5s.
- **Not byte-reproducible from raw files:** member refits (LightGBM draws, §1), CatBoost and Kumo on another GPU type (§1), the
  pseudo-label member refit inside the base line (`m_pseudolabel_pl3p88`, a LightGBM fit), and the two polars fold-side group-bys (§1).
  A run from raw files gives a file within that noise, not the submitted bytes.
- **Two lines, not one.** The submitted file combines the KCATLg line (`phase2/periods/final.env`, default seed bag) on the matched and
  other unmatched rows with the bag line (`phase2/periods/final_bag.env`, 24 + 16 seeds) on the unmatched-LIRF and exposed rows. One
  workspace with both settings gives a different file: the bag line's `final.v99` differs from the KCATLg line's on 353,021 matched rows
  (its base line was rerun from `np_derive`, including the pseudo-label member refit, and the unmatched rows enter the (B) day sums). The bag line was run with its lever scripts
  copied from a separate source directory (`P2_LEVER_SRC` of `phase2/steps/p2_v99_prep.py`); its replay from the iso stage onwards is
  byte-exact with the scripts of this repository, the steps before it were not rerun.
- **Constants not rebuilt from this repository.** The pool weights come from a 5-chain simplex fit over saved P12-generation fold vectors
  (F6, P12 dTK, Kq, RdLN, NNA3; `scripts/constants_2025.sh pool` reproduces the 4-decimal values .3212 / .3065 / .3722, the unrounded
  values to about 1e-5); a fit on the A2 chains of this repository gives .3338 / .3331 / .3331 and is not used. The 5-slot `pool_mix`
  call uses the fitted NNA3 weight 0.3722; the 3-weight lists of `pru_daily_calib.py` use 0.3723 so that the three weights sum to 1. The
  rebuilt post-rule fold frame `scratchpad/x4/X4_fold.parquet` is not byte-identical to the saved one; the fitted arms are identical.
- **Not published:** the KRS-NL LightGBM model (`scratchpad/xkrsnl/prod/gbm_A.txt`) and the 2025 unmatched-row context of F4
  (`scratchpad/xhr/unm2025b.parquet`) are fitted on or built from competition data; `make constants` writes both. The KRS β and KRS-NL
  coefficients are shipped in `phase2/params/` (`krs_coef.json`, `krsnl_coef.json`); TW1S (+1,264 s) is a code constant of
  `phase2/steps/p2_v107.sh`.
- **Pseudo-label fold teacher.** Its (B) vectors rebuild byte-exactly; the BA vectors differ by at most 1.1e-13 on 0.05 % of rows.
- **Pseudo-label teacher path.** The default teacher of `scripts/pseudo_labels.py prod` is `submissions/v99_iso_L.parquet`, the first-pass
  line of the workspace that `phase2/run_pipeline.sh` writes under that name and passes explicitly; it is a workspace stage file, not an
  upload.
