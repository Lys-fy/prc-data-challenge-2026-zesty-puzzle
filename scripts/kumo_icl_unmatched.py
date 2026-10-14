#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""F1 / F3: runs Kumo Tabular (in-context regressor) arms on the UNMATCHED rows, reusing the frame, MW weights, LOMO
nesting, gate G1-G7, placebo form and 2026 plausibility of the callsign-link lever (scripts/unm_link_lever.py)
unchanged. Inputs: positional subcommand (PLUMB | SMOKE | ARM [ARM ...] | SUMMARY | SEALCHECK), env
DEV/SIZE/NEST/CTX/SEED/NT/OUT below. Output: SEALCHECK writes the sealed column to a file; other subcommands print
gate numbers, computed on UNSEALED airport-days only (sealed rows in data/sealed_days.parquet are excluded and never printed).

Scopes (unm_other, non-LTFM; LTFM stays closed):
  UL = rows with a callsign link (V3's rows; chain = scratchpad/xur2/ARM_V3.npz cand)      UU = rows without a link (chain = frame.ref)
Arms (fixed before any number):
  B0 / B0s  V3 twin (LightGBM LOMO, seeds 0-2 / 3-5; callsign-link lever params; cand = ref + clip(g, +-1200))            [UL, controls]
  F1        Kumo medium ICL, context = the other 11 months' UL rows (all of them, <= CTX), A1 inputs (16 cols, NaN kept),
            target clip(y - ref, +-3600), g = mean of the 999 quantiles, cand = ref + clip(g, +-1200) (V3 form)   [UL, candidate]
  F1p       F1 with the clock block permuted within airport x month among UL rows (the placebo form A1p of unm_link_lever.py)  [UL, placebo]
  F2        0.5 * B0 + 0.5 * F1 (candidate-level blend)                                                     [UL, candidate]
  F3        Kumo ICL on UU rows, C1-type inputs (ref, apt, MS, VFR, hour, dow, stand timing, day state), target clip(y - ref, +-3600),
            cand = ref + 0.5 * clip(g, +-600) (the conservative A form of unm_link_lever.py: UU rows have no clock)          [UU, candidate]
  F3p       F3 with every input except (ref, apt, MS) permuted within airport x month                      [UU, placebo]
Gate (UNS rows; per scope): unm_link_lever.py G1-G5 vs chain, G6 beats placebo by >= 5 board s2 (4-fold mean), G7 twin rule (F vs B0 for UL: 4-fold mean < 0,
< 0 on >= 3/4, |mean| > max(2 |B0s - B0|, 5)), arm-count haircut: UL K = 2 -> 4-fold mean <= -(5 + 0.56 sigma), sigma = max(5, |B0s - B0|);
UU K = 1. 2026 plausibility Q1-Q3 as in unm_link_lever.py (Q2 only where an at-stand clock exists). Price = 4-fold MW(uns) mean / (2 * 228) s at 1x.
usage: kumo_icl_unmatched.py PLUMB | SMOKE | ARM [ARM ...] | SUMMARY | SEALCHECK        (ARM in B0 B0s F1 F1p F2 F3 F3p)
env: DEV (cuda|cpu), SIZE (medium), NEST (4), CTX (20000), SEED (0), PRC_THREADS, NT (lgb threads), OUT (scratchpad/xtfm2)
Run with $KUMO_PY for the Kumo arms; B0/B0s/SUMMARY/PLUMB need only .venv (lightgbm, polars)."""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import sys, os, time, json, types, numpy as np, polars as pl

ROOT = os.path.expanduser(os.environ.get("PRC_ROOT", os.getcwd()) + "")
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/scripts")
sys.path.insert(0, f"{ROOT}/scratchpad/xur")
t0 = time.time()
try:
    import lightgbm  # .venv-kumo has no lightgbm; unm_link_lever imports it at module level. The stub is only ever hit by B0/B0s (.venv).
except ImportError:
    sys.modules["lightgbm"] = types.ModuleType("lightgbm")
import unm_link_lever as X  # frame U, MW weights W, lab, score(), perm_within(), aptc(), col(); chdir $PRC_ROOT via unm_resid_lib
from unm_resid_lib import FOLDS, GATE, APTS, RN

OUT = os.environ.get("OUT", "scratchpad/xtfm2")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("DEV", "cuda")
SIZE = os.environ.get("SIZE", "medium")
NEST = int(os.environ.get("NEST", "4"))
CTX = int(os.environ.get("CTX", "20000"))
SEED = int(os.environ.get("SEED", "0"))
NT = int(os.environ.get("NT", "4"))
TAG = os.environ.get("TAG", "")
ECTX = int(os.environ.get("ECTX", "1"))
GTM = {}  # §8 twins: TAG suffix on saved arms, per-estimator context slices, trimmed-quantile readout (info)
ids, y, isr, mon, apt, ms, ref, W = X.ids, X.y, X.isr, X.mon, X.apt, X.ms, X.ref, X.W
uo, ltfm, fo = X.uo, X.ltfm, X.fo
lnk2 = X.U["lnk2"].to_numpy()
UL = uo & ~ltfm & lnk2
UU = uo & ~ltfm & ~lnk2
LAB_ALL = X.lab.copy()
# ---------------- sealed airport-days (data/sealed_days.parquet): removed from every scored set; never printed
SD = pl.read_parquet("data/sealed_days.parquet").select("ADEP_mvt", "day", "sealed")
Dk = X.U.select(pl.col("ADEP_mvt"), pl.col("MVT_TIME_UTC_mvt").dt.date().alias("day")).with_row_index("i")
Dk = Dk.join(SD, on=["ADEP_mvt", "day"], how="left").sort("i")
sealed = Dk["sealed"].fill_null(False).to_numpy().astype(bool)
miss = int(Dk["sealed"].is_null().to_numpy()[LAB_ALL].sum())
assert miss == 0, f"{miss} labelled 2025 rows without a sealed-day entry"
X.lab = LAB_ALL & ~sealed  # score() reads the module global `lab`
for f in FOLDS:  # rescale the MW weights to the full-fold weight sum, as sealed_score does for UNS
    k_all = LAB_ALL & (fo == f)
    k_uns = X.lab & (fo == f)
    if k_uns.any():
        X.W[k_uns] *= W[k_all].sum() / W[k_uns].sum()
W = X.W
lab = X.lab
V3 = np.load("scratchpad/xur2/ARM_V3.npz", allow_pickle=True)
assert np.array_equal(V3["ids"].astype(float), ids)
chainUL = V3["cand"].astype(float)


def chain_of(scope):
    return chainUL if scope == "UL" else ref.copy()


def rows_of(scope):
    return UL if scope == "UL" else UU


# ---------------- inputs
C = ["L_first_seen", "L_first_move", "L_hold_end", "L_at_stand", "L_land_prev"]


def X_A1():
    Lu = [X.col(c) for c in C]
    return np.column_stack(
        [ref, X.aptc(), ms] + Lu + [ref - c for c in Lu] + [X.col("n_ground"), X.col("d_first_stand"), X.col("dt_tko")]
    ).astype(np.float32)


C1_DAY = [
    "u_n",
    "u_n1h",
    "u_n2h",
    "u_ms_med",
    "u_ms_rank",
    "nbr_apt_m60_med",
    "nbr_apt_m60_max",
    "nbr_apt_m30_med",
    "arrx_off_day_med",
    "dreg_day_med",
    "dreg_day_late2h",
    "dreg_day_unm",
    "dreg_unm60",
]


def X_C1():
    Dm = np.column_stack([X.col(c) for c in C1_DAY])
    return np.column_stack(
        [
            ref,
            X.aptc(),
            ms,
            X.col("vfr"),
            X.col("hour"),
            X.col("dow"),
            X.col("stand_since_arr_inblock"),
            X.col("stand_since_prev_dep"),
            X.col("arr_block_delay"),
            Dm,
        ]
    ).astype(np.float32)


R = np.clip(y - ref, -3600, 3600)


# ---------------- learners (LOMO: 12 held-out months from the other 11; ranking rows from all 12)
def lomo_months(rows):
    for m in list(range(1, 13)) + ["rank"]:
        te = (rows & ~isr & (mon == m)) if m != "rank" else (rows & isr)
        tr = rows & LAB_ALL & ((mon != m) if m != "rank" else True)
        if te.sum():
            yield m, tr, te


def lomo_lgb(Xa, rows, seeds):
    import lightgbm as lgb

    PS = dict(
        objective="l2",
        learning_rate=0.03,
        num_leaves=15,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=20.0,
        verbose=-1,
        num_threads=NT,
    )
    g = np.full(len(y), np.nan)
    for m, tr, te in lomo_months(rows):
        p = np.zeros(te.sum())
        for s in seeds:
            p += lgb.train(dict(PS, seed=s), lgb.Dataset(Xa[tr], R[tr], categorical_feature=[1]), 300).predict(
                Xa[te]
            ) / len(seeds)
        g[te] = p
    return g


_model = None


def kumo_model():
    global _model
    if _model is None:
        import torch, sdm

        os.environ.setdefault("HF_HOME", f"{ROOT}/artifacts/xkumo/hf")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        torch.set_num_threads(int(os.environ.get("PRC_THREADS", "2")))
        _model = sdm.models.KumoTabular(task="regression", size=SIZE, device=DEV)
        print(
            f"kumo {SIZE} on {DEV} | torch {torch.__version__} sdm {getattr(sdm, '__version__', '?')} | nest {NEST} ctx {CTX} seed {SEED} | {time.time()-t0:.0f}s",
            flush=True,
        )
    return _model


def kumo_fit_predict(xc, yc, xq, chunk=4096, want_tm=False):
    """fit on (xc, yc) [per-estimator context slices when the pool exceeds CTX and ECTX], predict xq; returns mean of the 999 quantiles (and the 20 %-trimmed mean over quantiles if want_tm)."""
    import torch

    model = kumo_model()
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    ycf = yc.astype(np.float32)[:, None]
    if (
        ECTX and len(xc) > CTX
    ):  # NVIDIA adapter rule: consecutive CTX-slices of concatenated seeded permutations, one per estimator -> 3-D context, num_estimators=None
        rng = np.random.default_rng(SEED)
        perm = np.concatenate([rng.permutation(len(xc)) for _ in range(int(np.ceil(NEST * CTX / len(xc))))])[
            : NEST * CTX
        ].reshape(NEST, CTX)
        model.fit(
            torch.as_tensor(xc[perm], device=DEV),
            torch.as_tensor(ycf[perm], device=DEV),
            num_estimators=None,
            generator=gen,
        )
    else:
        model.fit(torch.as_tensor(xc, device=DEV), torch.as_tensor(ycf, device=DEV), num_estimators=NEST, generator=gen)
    out, otm = [], []
    for i in range(0, len(xq), chunk):
        p = model.predict(torch.as_tensor(xq[i : i + chunk], device=DEV))
        q = p.numerical if hasattr(p, "numerical") else p
        assert q.shape[-1] == 999, q.shape
        q = q.float()
        out.append(q.mean(-1).cpu().numpy())
        otm.append(torch.sort(q, dim=-1).values[:, 100:899].mean(-1).cpu().numpy())
    model.clear()
    return (np.concatenate(out), np.concatenate(otm)) if want_tm else np.concatenate(out)


def lomo_kumo(Xa, rows, tag):
    g = np.full(len(y), np.nan)
    gtm = np.full(len(y), np.nan)
    rng = np.random.default_rng(SEED)
    for m, tr, te in lomo_months(rows):
        ti = np.where(tr)[0]
        if len(ti) > CTX and not ECTX:
            ti = rng.choice(ti, CTX, replace=False)
        t1 = time.time()
        g[te], gtm[te] = kumo_fit_predict(Xa[ti], R[ti], Xa[te], want_tm=True)
        assert np.isfinite(g[te]).all(), (tag, m)
        print(
            f"  {tag}{TAG} month {m}: ctx {len(ti):,} query {te.sum():,} mean(g) {g[te].mean():+.1f} sd {g[te].std():.1f} {time.time()-t1:.0f}s",
            flush=True,
        )
    GTM[tag] = gtm
    return g


# ---------------- arms
def finish(cand, scope):
    rows = rows_of(scope)
    ch = chain_of(scope)
    cand = np.maximum(cand, 60.0)
    big = rows & (np.abs(cand - ref) > 7200)
    cand[big] = ref[big]
    out = ch.copy()
    out[rows] = cand[rows]
    return out


def arm(name):
    rng = np.random.default_rng(0)
    if name in ("B0", "B0s"):
        g = lomo_lgb(X_A1(), UL, (0, 1, 2) if name == "B0" else (3, 4, 5))
        cand = ref + 1.0 * np.clip(np.nan_to_num(g), -1200, 1200)
        return finish(cand, "UL"), g
    if name in ("F1", "F1p"):
        Xa = X_A1()
        if name == "F1p":
            Xa = np.column_stack([Xa[:, :3], X.perm_within(Xa[:, 3:], UL, rng)]).astype(np.float32)
        g = lomo_kumo(Xa, UL, name)
        cand = ref + 1.0 * np.clip(np.nan_to_num(g), -1200, 1200)
        return finish(cand, "UL"), g
    if name == "F2":
        zb, zf = (np.load(f"{OUT}/ARM_{a}.npz", allow_pickle=True) for a in ("B0", "F1"))
        cand = 0.5 * zb["cand"] + 0.5 * zf["cand"]
        return finish(cand, "UL"), 0.5 * (np.nan_to_num(zb["g"]) + np.nan_to_num(zf["g"]))
    if name in ("F3", "F3p"):
        Xa = X_C1()
        if name == "F3p":
            Xa = np.column_stack([Xa[:, :3], X.perm_within(Xa[:, 3:], UU, rng)]).astype(np.float32)
        g = lomo_kumo(Xa, UU, name)
        cand = ref + 0.5 * np.clip(np.nan_to_num(g), -600, 600)
        return finish(cand, "UU"), g
    raise SystemExit(f"unknown arm {name}")


SCOPE = {"B0": "UL", "B0s": "UL", "F1": "UL", "F1p": "UL", "F2": "UL", "F3": "UU", "F3p": "UU"}


def per_fold(cand, base, rows):
    d = (cand - y) ** 2 - (base - y) ** 2
    return {f: float((W[lab & rows & (fo == f)] * d[lab & rows & (fo == f)]).sum() / RN) for f in GATE}


def plaus(name, cand, g, scope):
    ch = chain_of(scope)
    rows = rows_of(scope)
    d = cand - ch
    ls = X.col("L_at_stand")
    anc = np.maximum(ls, 60.0)
    G = 1200.0 if scope == "UL" else 600.0
    res = {}
    for yr, s in (("2025", rows & ~isr & ~sealed), ("2026", rows & isr)):
        chg = s & (np.abs(d) >= 1)
        q1 = float(np.mean(np.abs(g[chg]) >= G)) if chg.any() else 0.0
        k2 = s & np.isfinite(ls) & (np.abs(d) >= 30)
        q2 = float(np.mean(np.sign(d[k2]) == np.sign(anc[k2] - ch[k2]))) if k2.any() else float("nan")
        res[yr] = (int(chg.sum()), q1, int(k2.sum()), q2, float(np.sqrt(np.mean(d[s] ** 2))) if s.any() else 0.0)
        print(
            f"  {name} {yr}: changed {res[yr][0]}, Q1 clip share {q1:.3f}; Q2 toward at-stand {q2:.3f} (n {k2.sum()}); RMS vs chain {res[yr][4]:.1f}",
            flush=True,
        )
    q2ok = (scope == "UU") or (res["2026"][2] == 0) or (res["2026"][3] >= 0.5)
    p = (res["2026"][1] <= 0.40) and q2ok and (res["2026"][4] <= 2 * max(res["2025"][4], 1.0))
    print(
        f"  PLAUS {name}: Q1 {res['2026'][1] <= 0.40} Q2 {q2ok} Q3 {res['2026'][4] <= 2 * max(res['2025'][4], 1.0)} -> {p}",
        flush=True,
    )
    return p


def summary():
    RES = {}
    for a in SCOPE:
        p = f"{OUT}/ARM_{a}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            RES[a] = dict(
                cand=z["cand"].astype(float),
                g=z["g"].astype(float),
                per=json.loads(str(z["per"])),
                g15=bool(z["g15"]),
                plaus=bool(z["plaus"]),
            )
    print(
        "\n######## KUMO_ICL_LIRF SUMMARY (UNS rows; 4-fold MW mean vs chain; per fold m39/m28/m410/janjul) "
        + time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    )
    for a, r in RES.items():
        m4 = np.mean(list(r["per"].values()))
        print(
            f"  {a:4s} [{SCOPE[a]}] {m4:+8.1f}  "
            + " ".join(f"{v:+7.1f}" for v in r["per"].values())
            + f"  G1-G5 {r['g15']} plaus {r['plaus']}"
        )

    def vs(a, b, scope):
        per = per_fold(RES[a]["cand"], RES[b]["cand"], rows_of(scope))
        return float(np.mean(list(per.values()))), per

    sig = 5.0
    if "B0" in RES and "B0s" in RES:
        s4, sp = vs("B0s", "B0", "UL")
        sig = max(5.0, abs(s4))
        print(
            f"  seed noise |B0s - B0| 4-fold {abs(s4):.1f} (per fold {', '.join(f'{v:+.1f}' for v in sp.values())}) -> sigma_ctrl {sig:.1f}"
        )
    verdict = {}
    for cand_arm, plc, K in (("F1", "F1p", 2), ("F2", "F1p", 2), ("F3", "F3p", 1)):
        if cand_arm not in RES:
            continue
        r = RES[cand_arm]
        m4 = float(np.mean(list(r["per"].values())))
        scope = SCOPE[cand_arm]
        g6 = (plc in RES) and (np.mean(list(RES[plc]["per"].values())) - m4 >= 5.0)
        if scope == "UL" and "B0" in RES:
            tm, tp = vs(cand_arm, "B0", "UL")
            g7 = tm < 0 and sum(v < 0 for v in tp.values()) >= 3 and abs(tm) > max(2 * (sig if sig > 5 else 0), 5.0)
            print(
                f"  {cand_arm} vs twin B0: 4-fold {tm:+.1f} (per fold {', '.join(f'{v:+.1f}' for v in tp.values())}) -> G7 {g7}"
            )
        else:
            g7 = True
        bar = -(5.0 + {1: 0.0, 2: 0.56, 3: 0.85}[K] * sig)
        hc = m4 <= bar
        ok = r["g15"] and g6 and g7 and hc and r["plaus"]
        verdict[cand_arm] = ok
        print(
            f"  {cand_arm}: 4-fold {m4:+.1f} | G1-G5 {r['g15']} | G6 placebo {g6} | G7 {g7} | haircut K={K} bar {bar:+.1f} {hc} | plaus {r['plaus']} => {'PASS' if ok else 'FAIL'} "
            f"(price at 1x ~ {m4/(2*228):+.3f} s)"
        )
    best = [a for a in ("F1", "F2") if verdict.get(a)]
    if len(best) == 2:
        best = [min(best, key=lambda a: np.mean(list(RES[a]["per"].values())))]
    print(f"  DECISION UL: {best[0] if best else 'none'} | UU: {'F3' if verdict.get('F3') else 'none'}")
    per_arm = ", ".join("%s: %+.1f" % (a, np.mean(list(r["per"].values()))) for a, r in RES.items())
    open(f"{OUT}/unm_summary.txt", "a").write(
        f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} UL {best} UU {verdict.get('F3')} per-arm {{{per_arm}}}\n"
    )


def sealcheck():
    """Sealed-row scorer, run once per experiment: sealed-row src_MW per fold for every stored arm vs its chain -> OUT/sealed_full.txt. Nothing printed."""
    lines = []
    for a in SCOPE:
        p = f"{OUT}/ARM_{a}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        cand = z["cand"].astype(float)
        rows = rows_of(SCOPE[a])
        base = chain_of(SCOPE[a])
        d = (cand - y) ** 2 - (base - y) ** 2
        per = {}
        for f in GATE:
            k = LAB_ALL & sealed & rows & (fo == f)
            ka = LAB_ALL & rows & (fo == f)
            sc = W[ka].sum() / max(W[k].sum(), 1e-9) if k.any() else 0.0
            per[f] = float((W[k] * d[k]).sum() / RN * sc) if k.any() else float("nan")
        lines.append(
            f"{a} [{SCOPE[a]}] SEAL src_MW per fold "
            + " ".join(f"{f} {v:+.1f}" for f, v in per.items())
            + f" | mean {np.nanmean(list(per.values())):+.1f} | rows {int((LAB_ALL & sealed & rows).sum())}"
        )
    open(f"{OUT}/sealed_full.txt", "w").write("\n".join(lines) + "\n")
    print(f"sealed column written to {OUT}/sealed_full.txt ({len(lines)} arms); not printed")


def twin(tags):
    """§8: twin{TAG} vs the incumbent arm (K = 1 per tag): G1-G5 + plaus of the twin vs chain AND twin - incumbent 4-fold MW(uns) mean <= -5. Info: trimmed-quantile readout."""
    V = {}
    for a, k, G in (("F1", 1.0, 1200.0), ("F3", 0.5, 600.0)):
        if not os.path.exists(f"{OUT}/ARM_{a}.npz"):
            continue
        inc = np.load(f"{OUT}/ARM_{a}.npz", allow_pickle=True)
        rows = rows_of(SCOPE[a])
        ch = chain_of(SCOPE[a])
        ci = inc["cand"].astype(float)
        print(f"\n== twins of {a} (incumbent 4-fold {np.mean(list(json.loads(str(inc['per'])).values())):+.1f})")
        for tg in tags:
            p = f"{OUT}/ARM_{a}{tg}.npz"
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            ct = z["cand"].astype(float)
            per = per_fold(ct, ci, rows)
            m4 = float(np.mean(list(per.values())))
            pv = json.loads(str(z["per"]))
            gtm = z["gtm"].astype(float)
            ctm = finish(ref + k * np.clip(np.nan_to_num(gtm), -G, G), SCOPE[a])
            ptm = per_fold(ctm, ci, rows)
            ok = bool(z["g15"]) and bool(z["plaus"]) and m4 <= -5.0
            V[a + tg] = dict(ok=ok, m4=m4, per=per)
            print(
                f"  {a}{tg} ({z['size']}, {int(z['nest'])} est): vs chain 4-fold {np.mean(list(pv.values())):+.1f} G1-G5 {bool(z['g15'])} plaus {bool(z['plaus'])} | vs incumbent {a}: per fold "
                + " ".join(f"{v:+.1f}" for v in per.values())
                + f" mean {m4:+.1f} (bar -5) => {'PASS' if ok else 'FAIL'} | info trimmed-quantile readout vs {a}: mean {np.mean(list(ptm.values())):+.1f}"
            )
    json.dump(V, open(f"{OUT}/twin_verdict.json", "w"))
    print("TWIN verdicts", {k: v["ok"] for k, v in V.items()})


def build2(base_path, base_md5):
    """§8 build on v99: replace the scope rows of each passing twin (best per scope if several); assert base rows = the incumbent arm's values."""
    import hashlib, shutil, subprocess

    assert hashlib.md5(open(base_path, "rb").read()).hexdigest().startswith(base_md5), "base md5"
    V = json.load(open(f"{OUT}/twin_verdict.json"))
    B = pl.read_parquet(base_path)
    base = B["TAXITIME_SEC_mvt"].to_numpy()
    pos = {v: i for i, v in enumerate(B["MVT_ID_mvt"].to_numpy().astype(float))}
    chosen = {}
    for a in ("F1", "F3"):
        c = [n for n, v in V.items() if n.startswith(a) and v["ok"]]
        if c:
            chosen[a] = min(c, key=lambda n: V[n]["m4"])
    if not chosen:
        print("no passing twin -> no build")
        return
    out = base.copy()
    sc = np.zeros(len(base), bool)
    tag = ""
    for a, n in chosen.items():
        inc = np.load(f"{OUT}/ARM_{a}.npz", allow_pickle=True)
        z = np.load(f"{OUT}/ARM_{n}.npz", allow_pickle=True)
        r = rows_of(SCOPE[a]) & isr
        ii = np.array([pos[v] for v in ids[r]])
        assert (
            np.maximum(np.rint(inc["cand"][r]), 60).astype(base.dtype) == base[ii]
        ).all(), f"{a}: base rows != incumbent arm"
        new = np.maximum(np.rint(z["cand"][r]), 60).astype(base.dtype)
        assert (np.abs(new.astype(float) - base[ii]) <= 7200).all()
        out[ii] = new
        sc[ii] = True
        tag += n
        d = new.astype(float) - base[ii]
        print(
            f"  {n}: {r.sum()} rows, changed {(d != 0).sum()}, mean {d[d != 0].mean() if (d != 0).any() else 0:+.1f}, RMS {np.sqrt((d ** 2).mean()):.1f}, max {np.abs(d).max():.0f}"
        )
    assert (out[~sc] == base[~sc]).all()
    p = f"stages/kumo_icl_twin{tag}.parquet"
    pl.DataFrame({"MVT_ID_mvt": B["MVT_ID_mvt"], "TAXITIME_SEC_mvt": out}).write_parquet(p)
    print(
        f"wrote {p} md5 {hashlib.md5(open(p, 'rb').read()).hexdigest()} | rows changed vs base {int((out != base).sum())}"
    )
    tmp = f"{OUT}/validate_check.parquet"
    shutil.copy(p, tmp)
    v = subprocess.run(
        [".venv/bin/python", "scripts/validate_submission.py", tmp, base_path], capture_output=True, text=True
    )
    print(v.stdout.strip(), v.stderr.strip()[-300:])
    os.remove(tmp)


if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"kumo_icl_unmatched: UL 2025 {int((UL & ~isr).sum())} (lab {int((UL & LAB_ALL).sum())}, unsealed {int((UL & lab).sum())}), 2026 {int((UL & isr).sum())} | "
        f"UU 2025 {int((UU & ~isr).sum())} (unsealed {int((UU & lab).sum())}), 2026 {int((UU & isr).sum())} | sealed share of labelled unm_other {sealed[LAB_ALL].mean():.3f} | {time.time()-t0:.0f}s",
        flush=True,
    )
    if args == ["PLUMB"]:
        Xa, Xc = X_A1(), X_C1()
        print(
            "A1 cols",
            Xa.shape,
            "finite share UL 2025 (clock block)",
            np.isfinite(Xa[UL & ~isr, 3:8]).mean(0).round(3),
            "| C1 cols",
            Xc.shape,
            "finite share UU 2025",
            np.isfinite(Xc[UU & ~isr]).mean(0).round(2),
        )
        print(
            "chain check: V3 cand == ref outside UL",
            bool(np.all(chainUL[~UL] == ref[~UL])),
            "| UL rows changed by V3 (2026)",
            int(((chainUL != ref) & UL & isr).sum()),
        )
        print("LOMO sizes UL:", {m: (int(tr.sum()), int(te.sum())) for m, tr, te in lomo_months(UL)})
        print("LOMO sizes UU:", {m: (int(tr.sum()), int(te.sum())) for m, tr, te in lomo_months(UU)})
        sys.exit(0)
    if args == ["SMOKE"]:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(3000, 16)).astype(np.float32)
        yy = 50 * x[:, 0] + 20 * x[:, 1] * x[:, 2] + rng.normal(0, 10, 3000)
        x[rng.random(x.shape) < 0.05] = np.nan
        t1 = time.time()
        g = kumo_fit_predict(x[:2000], yy[:2000], x[2000:])
        r = np.corrcoef(g, yy[2000:])[0, 1]
        print(
            f"SMOKE synthetic (5 % NaN): rmse {np.sqrt(((g - yy[2000:]) ** 2).mean()):.2f} (sd y {yy[2000:].std():.2f}) corr {r:.3f} {time.time()-t1:.1f}s"
        )
        assert r > 0.9
        sys.exit(0)
    if args == ["SUMMARY"]:
        summary()
        sys.exit(0)
    if args and args[0] == "TWIN":
        twin(args[1:])
        sys.exit(0)
    if args and args[0] == "BUILD2":
        build2(args[1], args[2])
        sys.exit(0)
    if args == ["SEALCHECK"]:
        sealcheck()
        sys.exit(0)
    for a in args:
        print(f"\n######## arm {a} [{SCOPE[a]}]", flush=True)
        cand, g = arm(a)
        scope = SCOPE[a]
        ch = chain_of(scope)
        m4, g15 = X.score(f"{a} vs chain ({'V3' if scope == 'UL' else 'ref'}), UNS rows", cand, base=ch)
        per = per_fold(cand, ch, rows_of(scope))
        pl_ok = plaus(a, cand, g, scope)
        if a == "B0":
            print(
                f"  B0 twin vs ARM_V3: max |diff| 2025 {np.nanmax(np.abs(cand - chainUL)[UL & ~isr]):.1f}, 2026 {np.nanmax(np.abs(cand - chainUL)[UL & isr]):.1f} (host {os.uname().machine})"
            )
        np.savez_compressed(
            f"{OUT}/ARM_{a}{TAG}.npz",
            ids=ids,
            cand=cand,
            g=g,
            gtm=GTM.get(a, np.full(len(y), np.nan)),
            isr=isr,
            uo=uo,
            ul=UL,
            uu=UU,
            chain=ch,
            per=json.dumps(per),
            g15=g15,
            plaus=pl_ok,
            mean4=m4,
            size=SIZE,
            nest=NEST,
        )
        print(
            f"  saved {OUT}/ARM_{a}{TAG}.npz ({SIZE}, {NEST} est) | 4-fold MW(uns) mean {m4:+.1f} | {time.time()-t0:.0f}s",
            flush=True,
        )
    print("done", f"{time.time()-t0:.0f}s")
