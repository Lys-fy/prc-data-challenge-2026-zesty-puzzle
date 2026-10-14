#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Schedule-copy rule at LIRF, apply step (the last stage): write the rule prices onto a submission file.
Inputs: <base parquet> (MVT_ID_mvt, TAXITIME_SEC_mvt), <final_dryrun.parquet> of scripts/rule_schedule_copy_lirf_relink.py (new_int).
Output: <out parquet>; only the rule rows change, every other row stays byte-identical to the base, in the same order.
usage: rule_schedule_copy_lirf_apply.py <base parquet> <final_dryrun.parquet> <out parquet>      Refuses to overwrite.

No LOBT clip is needed: the rule rows are unmatched (no NM record, so no mvt_lobt); the validator checks the LOBT law anyway.
"""

import os, sys, hashlib, numpy as np, polars as pl


def apply_rule(B, D):
    """B: submission frame; D: relink output (MVT_ID_mvt, mon, new_int). Returns (O, D'): O = B with new_int on the rule
    rows, D' = D as (MVT_ID_mvt Float64, mon, xml Int32). Asserts unique rule ids present in B, prices >= 60 s, B's order and schema.
    """
    D = D.select(MVT_ID_mvt=pl.col("MVT_ID_mvt").cast(pl.Float64), mon="mon", xml=pl.col("new_int").cast(pl.Int32))
    assert D.height == D["MVT_ID_mvt"].n_unique() and D["xml"].null_count() == 0 and D["xml"].min() >= 60
    assert D.join(B, on="MVT_ID_mvt", how="anti").height == 0, "schedule_copy_rule ids missing from the base"
    O = (
        B.join(D, on="MVT_ID_mvt", how="left", maintain_order="left")
        .with_columns(TAXITIME_SEC_mvt=pl.coalesce("xml", "TAXITIME_SEC_mvt"))
        .select(B.columns)
    )
    assert (
        O.schema == B.schema and O.height == B.height and (O["MVT_ID_mvt"] == B["MVT_ID_mvt"]).all()
    ), "order/schema changed"
    assert O["TAXITIME_SEC_mvt"].null_count() == 0 and O["TAXITIME_SEC_mvt"].min() >= 60
    return O, D


def main(argv):
    if argv[:1] in (["-h"], ["--help"]):
        sys.exit(print(__doc__))
    base_p, dry_p, out_p = argv[:3]
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    assert not os.path.exists(out_p), f"refusing: {out_p} exists"
    B = pl.read_parquet(base_p)
    D = pl.read_parquet(dry_p)
    assert (
        B.schema == {"MVT_ID_mvt": pl.Float64, "TAXITIME_SEC_mvt": pl.Int32} and B["MVT_ID_mvt"].n_unique() == B.height
    ), B.schema
    O, D = apply_rule(B, D)
    ch = O["TAXITIME_SEC_mvt"] != B["TAXITIME_SEC_mvt"]
    C = B.with_columns(new=O["TAXITIME_SEC_mvt"]).filter(ch).join(D, on="MVT_ID_mvt", how="left")
    assert C["xml"].null_count() == 0, "a non-schedule_copy_rule row changed"
    print(
        f"schedule_copy_rule rows {D.height} by month {dict(sorted(D.group_by('mon').len().iter_rows()))}; changed {C.height} by month {dict(sorted(C.group_by('mon').len().iter_rows()))}"
        f" (schedule_copy_rule rows equal to base already: {D.height - C.height})"
    )
    d = (C["new"] - C["TAXITIME_SEC_mvt"]).cast(pl.Float64).to_numpy()
    print(
        f"change: mean {d.mean():+.1f} s, RMS {np.sqrt((d ** 2).mean()):.1f} s, max |d| {np.abs(d).max():.0f} s; old mean {C['TAXITIME_SEC_mvt'].mean():.0f} -> new {C['new'].mean():.0f}"
    )
    tmp = out_p + ".tmp"
    O.write_parquet(tmp)
    os.link(tmp, out_p)
    os.unlink(tmp)  # os.link fails if out_p appeared meanwhile
    print(f"wrote {out_p}: rows {O.height:,}, md5 {md5(out_p)} (base md5 {md5(base_p)})")


if __name__ == "__main__":
    main(sys.argv[1:])
