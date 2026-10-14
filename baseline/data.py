# SPDX-License-Identifier: GPL-3.0-only
"""Load the competition parquet files and add epoch-second time columns."""
from pathlib import Path

import polars as pl

TIME_COLS = {"t": "MVT_TIME_UTC_mvt", "tb": "BLOCK_TIME_UTC_mvt", "ts": "SCHED_TIME_UTC_mvt", "ta": "AOBT_3_flt",
             "tl": "LOBT_flt", "te": "EOBT_1_flt", "ti": "IOBT_flt"}


def training_files(raw_dir: Path) -> list[Path]:
    """The twelve 2025 training files in date order."""
    files = sorted(raw_dir.glob("training_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no training_*.parquet in {raw_dir} (run `make data`)")
    return files


def load(files: list[Path]) -> pl.DataFrame:
    """All rows (departures and arrivals) of the given files, with epoch seconds t, tb, ts, ta, tl, te, ti and the month."""
    rows = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    epochs = [pl.col(c).dt.epoch("s").cast(pl.Float64).alias(k) for k, c in TIME_COLS.items()]
    return rows.with_columns(*epochs, month=pl.col("MVT_TIME_UTC_mvt").dt.month(),
                             is_dep=pl.col("PHASE_mvt") == "DEP", matched=pl.col("FLIGHT_ID_mvt").is_not_null())
