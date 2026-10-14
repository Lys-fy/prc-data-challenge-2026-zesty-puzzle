#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Label-free airport-day NOVELTY score = Mahalanobis distance of the day's mean standardised token features to the airport's 2025 days
(the member's numeric matrix after clip / impute / standardise on 2025 rows; day means projected on the top-20 pooled PCs; per-airport mean/cov with shrinkage
0.1 to the diagonal). Threshold per airport = the 2025 p99. Reports the 2026 shares (days, rows) above it per airport, and re-runs the (i)
RMS-ratio check of D = big_ctx3 - big_lgb (2026 prod rows vs 2025 janjul held rows) with the novelty mask instead of the EHAM mask.
Writes scratchpad/xds2/novelty_days.parquet (tiny) and scratchpad/xds2/novelty_mask_ids.npy (MVT_IDs of flagged rows, 2025 + 2026). No label is read."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import os, sys, numpy as np, polars as pl
os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, f"{ROOT}/scripts")
from train2 import CAT_BASE, cols_for
NPC, SHR, Q = 20, 0.1, 0.99; O = "scratchpad/xds2"; os.makedirs(O, exist_ok=True)
P = "data/features_v9.parquet"; S = pl.read_parquet_schema(P); dummy = pl.DataFrame(schema={c: pl.Null for c in S})
LEAK = {"y", "y_minus_ref", "fb", "fb_pru", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "t_block", "is_rank", "m", "MVT_ID_mvt"}
num = []
for g in "base,tl,ref,turn,te,wx,arrfb,nbr,te2,opdi,day".split(","): num += cols_for(dummy, g)
NUMT = (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt16, pl.UInt8, pl.Boolean)
num = [c for c in dict.fromkeys(num) if c in S and c not in LEAK and c not in CAT_BASE and S[c] in NUMT]
d = pl.read_parquet(P, columns=list(dict.fromkeys(["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "m", "is_rank", "mvt_aobt3"] + num)))
d = d.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.date().alias("_day")).sort(["ADEP_mvt", "_day", "MVT_TIME_UTC_mvt"])
is26 = d["is_rank"].to_numpy() == 1; ids = d["MVT_ID_mvt"].to_numpy(); apt = d["ADEP_mvt"].cast(pl.Utf8).to_numpy(); day = d["_day"].to_numpy(); mon = d["m"].to_numpy()
X = d.select([pl.col(c).cast(pl.Float32) for c in num]).to_numpy().astype(np.float32); X[~np.isfinite(X)] = np.nan; offna = np.isnan(d["mvt_aobt3"].to_numpy().astype(float))
ref = ~is26; rng = np.random.default_rng(0); sub = rng.choice(np.where(ref)[0], 300000, replace=False)
na = np.isnan(X); flag_cols = np.where(na[sub].mean(0) > 0.01)[0]
lo = np.nan_to_num(np.nanpercentile(X[sub], 0.5, axis=0)); hi = np.nan_to_num(np.nanpercentile(X[sub], 99.5, axis=0))
X = np.clip(X, lo, hi); med = np.nan_to_num(np.nanmedian(X[sub], axis=0)); X = np.where(na, med, X); mu = X[sub].mean(0); sd = X[sub].std(0); sd[sd < 1e-6] = 1; X = (X - mu) / sd
X = np.concatenate([X, na[:, flag_cols].astype(np.float32), offna[:, None].astype(np.float32)], 1); del na
dk = np.array([f"{a}|{b}" for a, b in zip(apt, day)]); days, start = np.unique(dk, return_index=True); o = np.argsort(start); start = start[o]; days = days[o]; end = np.append(start[1:], len(dk))
M = np.vstack([X[s:e].mean(0) for s, e in zip(start, end)]); del X
d_apt = apt[start]; d_26 = is26[start]; d_n = end - start; d_date = day[start]; d_m = mon[start]
print(f"airport-days {len(start):,} (2026 {int(d_26.sum())}) | token features {M.shape[1]}", flush=True)
# pooled PCA on 2025 days
R = M[~d_26]; c0 = R.mean(0); U, s, Vt = np.linalg.svd(R - c0, full_matrices=False); W = Vt[:NPC].T; Z = (M - c0) @ W
print(f"PCA top-{NPC} explained {float((s[:NPC] ** 2).sum() / (s ** 2).sum()):.3f}", flush=True)
d2 = np.full(len(start), np.nan); thr = {}
for A in sorted(set(d_apt)):
    r25 = (d_apt == A) & ~d_26; z = Z[r25]; mu_a = z.mean(0); Cv = np.cov(z.T); Cv = (1 - SHR) * Cv + SHR * np.diag(np.diag(Cv)); Ci = np.linalg.inv(Cv)
    ia = d_apt == A; dz = Z[ia] - mu_a; d2[ia] = np.einsum("ij,jk,ik->i", dz, Ci, dz); thr[A] = float(np.quantile(d2[r25], Q))
flag = np.array([d2[i] > thr[d_apt[i]] for i in range(len(start))])
T = pl.DataFrame({"apt": d_apt, "date": d_date, "n": d_n, "m": d_m, "is2026": d_26, "d2": d2, "p99_2025": [thr[a] for a in d_apt], "flag": flag}); T.write_parquet(f"{O}/novelty_days.parquet")
print("## 2026 novelty per airport (share of days / rows above the airport's 2025 p99; 2025 in-sample share by construction ~1 %)")
for A in sorted(thr):
    ia = (d_apt == A); a26 = ia & d_26; a25 = ia & ~d_26
    print(f"  {A}: 2026 days {int(a26.sum())} flagged {int(flag[a26].sum())} ({flag[a26].mean():.2%}) rows {d_n[a26 & flag].sum() / max(1, d_n[a26].sum()):.2%} | 2025 flagged rows {d_n[a25 & flag].sum() / d_n[a25].sum():.2%} | median d2 2026/2025 {np.median(d2[a26]) / np.median(d2[a25]):.2f}")
print(f"## TOTAL 2026 rows flagged {d_n[d_26 & flag].sum() / d_n[d_26].sum():.2%} (rule <= 10 %) | 2025 rows flagged {d_n[~d_26 & flag].sum() / d_n[~d_26].sum():.2%} (held rule <= 3 %)")
e = (d_apt == "EHAM") & d_26
print("## EHAM 2026 days: ", " ".join(f"{str(dt)[5:]}:{v:.0f}{'*' if f else ''}" for dt, v, f in zip(d_date[e], d2[e], flag[e])))
mask_ids = np.concatenate([ids[s:e] for s, e, f in zip(start, end, flag) if f]); np.save(f"{O}/novelty_mask_ids.npy", mask_ids); print(f"mask ids {len(mask_ids):,}")
# §6c (i) re-run with the novelty mask instead of the EHAM mask: D = CTX3 - LGB
def Dv(a, b):
    A, B = np.load(a), np.load(b); oa, ob = np.argsort(A[0]), np.argsort(B[0]); assert np.array_equal(A[0][oa], B[0][ob]); return A[0][oa], A[1][oa] - B[1][ob]
P3 = np.mean([np.load(f"data/oof/big_ctx{s}_prod.npy")[1] for s in ("", "_s1", "_s2")], 0); i0 = np.load("data/oof/big_ctx_prod.npy")[0]
for k, s in enumerate(("", "_s1", "_s2")): assert np.array_equal(np.load(f"data/oof/big_ctx{s}_prod.npy")[0], i0)
np.save(f"{O}/tmp_CTX3_prod.npy", np.vstack([i0, P3, np.full(len(i0), np.nan)]))
i25, D25 = Dv("data/oof/big_ctx3_janjul.npy", "data/oof/big_lgb_janjul.npy"); i26, D26 = Dv(f"{O}/tmp_CTX3_prod.npy", "data/oof/big_lgb_prod.npy"); os.remove(f"{O}/tmp_CTX3_prod.npy")
ap_of = dict(zip(ids, apt)); fl = set(mask_ids.tolist())
a25 = np.array([ap_of[i] for i in i25]); a26 = np.array([ap_of[i] for i in i26]); k25 = np.array([i not in fl for i in i25]); k26 = np.array([i not in fl for i in i26])
r = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else np.nan
print(f"## §6c(i) with novelty mask: RMS ratio overall {r(D26[k26]) / r(D25[k25]):.2f} (unmasked {r(D26) / r(D25):.2f})")
ok = r(D26[k26]) / r(D25[k25]) <= 1.5
for A in sorted(thr):
    m25, m26 = a25 == A, a26 == A; rr = r(D26[m26 & k26]) / r(D25[m25 & k25]); ru = r(D26[m26]) / r(D25[m25]); ok &= rr <= 1.5
    print(f"  {A}: ratio masked {rr:.2f} (unmasked {ru:.2f}) {'ok' if rr <= 1.5 else 'FLAG'}")
sh26 = d_n[d_26 & flag].sum() / d_n[d_26].sum(); sh25 = 1 - k25.mean()
print("## §4 eligibility:", "PASS" if (ok and sh26 <= 0.10 and sh25 <= 0.03) else "FAIL", f"(ratios ok {ok}, 2026 masked {sh26:.2%}, 2025 janjul held masked {sh25:.2%})")
