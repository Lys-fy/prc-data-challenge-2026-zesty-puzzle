#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""phase2_ws period-input helper. --check reports, per open-data input, the coverage share of the period's ranking
departures, writes report/period_inputs.json, and prints which fallback applies. --fetch-adsb pulls and extracts
adsb.lol days into the workspace. --build-adsb rebuilds the ADS-B clock feature tables from the extracted days.
Inputs: data/features_v9.parquet plus the external PRU/adsb.lol/METAR/OPDI files named below, and day args for
--fetch-adsb. Outputs: report/period_inputs.json (--check) or updated data/external/* tables (the other two modes).

phase2_ws: period-dependent open-data inputs of the phase-2 pipeline, with coverage report and graceful fallback.
  --check        per input, the share of the period's ranking departures covered; writes report/period_inputs.json and prints the
                 fallback that applies (the run_pipeline flags PRU / ADSB in the period file).
  --fetch-adsb D1 [D2 ...]   pull + extract adsb.lol days into this workspace (scripts/adsb_pull.py, adsb_extract.py, adsb_extract2.py,
                 adsb_extract3.py, and for the v83+ clock stages adsb_pushstart_extract.py (adsb6 push-start clock, clkP_*) and adsb_longstop_extract.py
                 (adsbL longest-stop clock, clkL_*), i.e. the first half of the ADS-B table rebuild for new days.
                 Days are YYYY-MM-DD; '--days <file>' reads them from a file.
  --build-adsb   rebuild the per-row clock tables from all extracted days: build_adsb_features.py, build_adsb2/3/5_features.py and
                 build_adsb6_features.py, build_adsbL_features.py (the v87+ chain reads adsb6 and adsbL; without them those clocks are null).
Inputs and fallbacks:
  PRU daily (EUROCONTROL/PRU airport daily pre-departure delay; data/external/xpr_pru_daily.parquet built by scripts/pru_build.py):
      missing days -> period file PRU=0: chains without the xpr meta inputs (the v68 'bundle2' form) and no (B) calibration.
  adsb.lol (ODbL; data/external/adsblol/apt_YYYYMMDD.parquet -> adsb/adsb2/adsb3/adsb5 tables): a day without data leaves its rows
      unlinked (adsb_linked = 0). Every stage already routes unlinked rows to its clock-free branch (dark-row input xdk, clock stages
      skip them), so no code change is needed: the fallback is by construction (as LIRF July 2026, 0 % linked).
  METAR (Iowa Environmental Mesonet ASOS; wx* columns of features_v9 via scripts/metar_features.py): missing hours -> NaN wx features,
      handled natively by the tree members (as the 2025 gaps).
  OPDI (EUROCONTROL Open Performance Data Initiative v0.0.2; data/external/opdi/flight_list_YYYYMM.parquet -> opdi_features.parquet, opdi_*
      columns): the release ends in July 2026. A period after that has all-null opdi_* columns although every member since v13 was trained
      with OPDI present: NO fallback exists (an OPDI-free member set would need retraining). The check reports the months without a
      flight list and the share of ranking departures with an OPDI link."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, json, glob, subprocess, numpy as np, polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")


def check():
    F = pl.read_parquet(
        "data/features_v9.parquet",
        columns=["MVT_ID_mvt", "is_rank", "ADEP_mvt", "MVT_TIME_UTC_mvt", "unmatched", "wx0_sknt"],
    ).filter(pl.col("is_rank") == 1)
    F = F.with_columns(ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8), day=pl.col("MVT_TIME_UTC_mvt").dt.date())
    days = sorted(F["day"].unique().to_list())
    rep = {"rank_rows": F.height, "days": [str(days[0]), str(days[-1])], "n_days": len(days)}
    R = pl.read_parquet("data/external/xpr_pru_daily.parquet").with_columns(ADEP_mvt=pl.col("ADEP_mvt").cast(pl.Utf8))
    j = F.join(R.select("ADEP_mvt", "day", "allpre_min"), on=["ADEP_mvt", "day"], how="left")
    rep["pru_cover"] = float(j["allpre_min"].is_not_null().mean())
    A = pl.read_parquet("data/external/adsb_features.parquet", columns=["MVT_ID_mvt", "adsb_linked"])
    ja = F.join(A, on="MVT_ID_mvt", how="left")
    rep["adsb_row_in_table"] = float(ja["adsb_linked"].is_not_null().mean())
    rep["adsb_linked"] = float((ja["adsb_linked"].fill_null(0) == 1).mean())
    have = {os.path.basename(p)[4:12] for p in glob.glob("data/external/adsblol/apt_*.parquet")}
    rep["adsb_days_missing"] = [str(d) for d in days if d.strftime("%Y%m%d") not in have]
    rep["metar_cover"] = float(F["wx0_sknt"].is_not_null().mean())
    months = sorted({d.strftime("%Y%m") for d in days})
    have_m = {os.path.basename(p)[12:18] for p in glob.glob("data/external/opdi/flight_list_*.parquet")}
    rep["opdi_months_missing"] = [m for m in months if m not in have_m]
    if os.path.exists("data/external/opdi_features.parquet"):
        O = pl.read_parquet("data/external/opdi_features.parquet", columns=["MVT_ID_mvt", "opdi_linked"]).with_columns(
            pl.col("MVT_ID_mvt").cast(F["MVT_ID_mvt"].dtype)
        )
        jo = F.select("MVT_ID_mvt").join(O, on="MVT_ID_mvt", how="left")
        rep["opdi_row_in_table"] = float(jo["opdi_linked"].is_not_null().mean())
        rep["opdi_linked"] = float((jo["opdi_linked"].fill_null(0) == 1).mean())
    else:
        rep["opdi_row_in_table"] = rep["opdi_linked"] = 0.0
    rep["by_airport"] = {
        a: {
            "pru": float(g["allpre_min"].is_not_null().mean()),
            "adsb_linked": float((g["adsb_linked"].fill_null(0) == 1).mean()),
        }
        for (a,), g in ja.join(j.select("MVT_ID_mvt", "allpre_min"), on="MVT_ID_mvt").group_by(["ADEP_mvt"])
    }  # key tuple (polars >= 1)
    os.makedirs("report", exist_ok=True)
    json.dump(rep, open("report/period_inputs.json", "w"), indent=1)
    print(f"period {rep['days'][0]}..{rep['days'][1]} ({rep['n_days']} days, {rep['rank_rows']:,} ranking departures)")
    print(
        f"  PRU daily cover {rep['pru_cover']:.3f} -> {'PRU=1 ok' if rep['pru_cover'] > 0.99 else 'set PRU=0 (fallback: bundle2 chains, no (B)) or rebuild xpr_pru_daily with pru_build.py'}"
    )
    print(
        f"  ADS-B: rows in clock table {rep['adsb_row_in_table']:.3f}, linked {rep['adsb_linked']:.3f}, days without a pulled archive {len(rep['adsb_days_missing'])}"
        + (
            " -> unlinked rows use the clock-free branches (no action needed); pull with --fetch-adsb to add them"
            if rep["adsb_days_missing"]
            else ""
        )
    )
    print(f"  METAR cover {rep['metar_cover']:.3f}")
    print(
        f"  OPDI: rows in opdi_features {rep['opdi_row_in_table']:.3f}, linked {rep['opdi_linked']:.3f}, months without a flight list {rep['opdi_months_missing'] or 'none'}"
        + (
            " -> NO FALLBACK: opdi_* null on these days, members were trained with OPDI present (expect degraded members; OPDI v0.0.2 ends Jul 2026)"
            if rep["opdi_months_missing"] or rep["opdi_linked"] < 0.5
            else " ok"
        )
    )
    for a in sorted(rep["by_airport"]):
        print(f"    {a} PRU {rep['by_airport'][a]['pru']:.3f} linked {rep['by_airport'][a]['adsb_linked']:.3f}")


def fetch_adsb(days):
    py = sys.executable
    if days[:1] == ["--days"]:
        days = open(days[1]).read().split()
    # adsb6 (adsb_pushstart_extract.py, v83 push-start clock) and adsbL (adsb_longstop_extract.py, v87 longest-stop clock) read the same apt_/clk_ day files
    for s in (
        "adsb_pull.py",
        "adsb_extract.py",
        "adsb_extract2.py",
        "adsb_extract3.py",
        "adsb_pushstart_extract.py",
        "adsb_longstop_extract.py",
    ):
        r = subprocess.run([py, f"scripts/{s}"] + days)
        print(s, "rc", r.returncode, flush=True)
        if r.returncode:
            sys.exit(r.returncode)


def build_adsb():
    py = sys.executable
    for s in (
        "build_adsb_features.py",
        "build_adsb2_features.py",
        "build_adsb3_features.py",
        "build_adsb5_features.py",
        "build_adsb6_features.py",
        "build_adsbL_features.py",
    ):
        r = subprocess.run([py, f"scripts/{s}"])
        print(s, "rc", r.returncode, flush=True)
        if r.returncode:
            sys.exit(r.returncode)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check()
    elif sys.argv[1] == "--fetch-adsb":
        fetch_adsb(sys.argv[2:])
    elif sys.argv[1] == "--build-adsb":
        build_adsb()
