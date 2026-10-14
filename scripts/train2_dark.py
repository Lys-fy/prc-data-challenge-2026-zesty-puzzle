#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Trainer of the airport-dedicated members lgb_airport_<APT> (--only-apt, --apt-weight).
--unlinked-only: train and validate on adsb_linked != 1 matched rows only.
--target resid_clock: offset = a label-free clock anchor T_hat, priority adsb2_leave > adsb3_mv_gs > adsb_move > AOBT_3, each clock only
inside [--clk-lo, --clk-hi] (needs feats adsb,adsb2,adsb3); adds the features xlk_src / xlk_anchor / xlk_a3d.
--linked-only: train and validate on adsb_linked == 1 matched rows only; ranking predictions are written for all ranking rows and used on
linked rows only.
LightGBM / CatBoost mixture-member trainer on the features table (default data/features_v2.parquet; --features data/features_v9.parquet in the recipes).
--feats      comma list of feature groups (default base,tl,ref,turn,te)
--strategy   keep | cleanpru | mixture   (mixture = P(artefact)*(MVT-SCHED) + (1-P)*clean-regressor)
--cv         janjul (train other months -> val Jan+Jul, weighted 44/56) | forward (train Jan-Oct -> val Nov+Dec) | months:M1,M2
--model      lgbm | cat | xgb
Writes a JSON line to --out, OOF validation predictions to data/oof/<name>.npy, ranking predictions with --predict-rank."""
"""Further targets: --target mov | rep."""
import argparse, json, time, os
from pathlib import Path
import numpy as np, polars as pl, lightgbm as lgb, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
CAT_BASE = ["ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "apt_rwy", "apt_stand", "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt",
            "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt", "apt_unm", "cs_pre", "FLIGHT_RULE_mvt"]
G = {
 "base": ["hour", "dow", "doy", "sched_min", "mvt_aobt3", "mvt_lobt", "mvt_eobt1", "mvt_iobt", "mvt_sched", "eobt_sched", "aobt3_eobt1",
          "aobt3_sched", "blocktime3", "blocktime1", "diverted", "cs_len", "cs_airline_style", "unmatched"],
 "tl": None, "ref": None, "turn": ["stand_since_arr_inblock", "stand_since_arr_land", "arr_block_delay", "stand_since_prev_dep"],
 "te": None, "geo": None, "wx": None, "day": None, "dst": None, "arrfb": None, "nbr": None, "te2": None, "opdi": None, "nbrx": None, "arrx": None, "opdir": None, "te3": None, "atfm": None, "ert": None, "storm": None, "reg": None, "dur": None, "sarr": None, "durp": None, "fbte": None, "dreg": None, "dreg2": None, "areg": None, "rreg": None, "dreg3": None, "dreg3perm": None, "preg": None, "wxs": None, "nbrs": None, "nbrsc": None, "adsb": None, "adsb2": None, "adsbF": None, "adsb3": None, "adsb4": None, "adsbU": None, "arrt": None, "xe": None, "xb39": None, "xb28": None, "xb410": None, "xb17": None, "xbP": None, "xba": None,   # XB flight memory (per-fold side tables) + AOBT_3-anchored stand arrivals
}
def cols_for(df, group):
    if G.get(group): return [c for c in G[group] if c in df.columns]
    pref = {"tl": ("rwy_dep_", "apt_dep_", "apt_arr_", "rwy_arr_", "rwy_heavy_", "sched_dep_", "eobt_dep_", "gap_prev", "gap_next", "queue_run", "run_len", "queue_pos",
                   "prev_heavy", "prev2_heavy", "n_rwy_in_hour", "rwy_share_hr", "rwy_hr_n", "apt_hr_n"),
            "ref": ("ref_sr_", "ref_s_", "ref_r_", "ref_sh_", "aobt3_minus_ref"), "te": ("te_op_", "te_fno_", "te_cs_", "te_des_"),
            "geo": ("dist_", "n_turns", "path_"), "day": ("day_", "hr_ndep", "hr_narr", "hr_dep_ratio"), "dst": ("dst_",), "arrfb": ("arrfb_", "arrart_", "arr_day_n", "arr_dayop_n", "arrlate_dayop_n"), "wx": ("wx0_", "wx30_", "wx60_", "wxd_", "wx_age"), "nbr": ("nbr_",), "te2": ("te2_",), "opdi": ("opdi_",), "nbrx": ("nbrx_",), "arrx": ("arrx_",), "opdir": ("opdir_",), "te3": ("te3_",), "atfm": ("atfm_",), "ert": ("ert_",), "storm": ("storm_",), "reg": ("reg_",), "dur": ("dur_",), "sarr": ("sarr_",), "durp": ("durp_",), "fbte": ("fbte_",), "dreg": ("dreg_",), "dreg2": ("dreg2_",), "areg": ("areg_",), "rreg": ("rreg_",), "dreg3": ("dreg3_",), "dreg3perm": ("dreg3perm_",), "preg": ("preg_",), "wxs": ("wxs_",), "nbrs": ("nbrs_",), "nbrsc": ("nbrsc_",), "adsb": ("adsb_",), "adsb2": ("adsb2_",), "adsbF": ("adsbF_",), "adsb3": ("adsb3_",), "adsb4": ("adsb4_",), "adsbU": ("adsbU_",), "arrt": ("arrt_",), "xe": ("xe_",), "xb39": ("xb_",), "xb28": ("xb_",), "xb410": ("xb_",), "xb17": ("xb_",), "xbP": ("xb_",), "xba": ("xba_",)}[group]
    return [c for c in df.columns if c.startswith(pref)]

def dayshift_model(trn_mask, unm, adep, ms, y, min_events=5):
    """Per-airport logistic P(day-shifted label | MVT-SCHED hours) on unmatched training departures with MVT-SCHED > 8 h. Returns {airport: (a, b)} for p = sigmoid(a + b*h)."""
    from sklearn.linear_model import LogisticRegression
    models = {}
    for a_ in np.unique(adep):
        c_ = trn_mask & unm & (adep == a_) & (ms > 28800); lab = (y[c_] > 80000).astype(int)
        if c_.sum() >= 10 and lab.sum() >= min_events and lab.sum() < len(lab):
            lr = LogisticRegression(C=1.0).fit((ms[c_] / 3600.0).reshape(-1, 1), lab); models[a_] = (float(lr.intercept_[0]), float(lr.coef_[0][0]))
    return models
TAIL_KEYS = ["lr", "leaves", "min_leaf", "ff", "l2", "cat_smooth", "rounds", "stake_power", "p_power", "stake_floor"]
def parse_tail(spec):
    v = [float(x) for x in spec.split(",")]; assert len(v) == len(TAIL_KEYS), f"--cls-tail needs {len(TAIL_KEYS)} values: {','.join(TAIL_KEYS)}"
    return dict(zip(TAIL_KEYS, v))

def dayshift_model_pooled(trn_mask, unm, adep, ms, y, tau=1.0):
    """Hierarchical logistic P(day-shift | MVT-SCHED hours): shared slope b, per-airport intercepts a_apt shrunk to a global a_0 with a Gaussian
    prior of sd tau (MAP by scipy). Fitted on unmatched training departures with MVT-SCHED > 8 h. Airports without events get a small,
    delay-dependent probability instead of a hard zero. Returns {airport: (a_apt, b)}."""
    from scipy.optimize import minimize
    c_ = trn_mask & unm & (ms > 28800); h = ms[c_] / 3600.0; lab = (y[c_] > 80000).astype(float); ap_ = adep[c_]
    apts = sorted(set(adep)); idx = np.array([apts.index(a_) for a_ in ap_])
    def nll(theta):
        a0, b = theta[0], theta[1]; a = theta[2:]; z = a0 + a[idx] + b * h; p = 1 / (1 + np.exp(-z))
        eps = 1e-9; return -np.sum(lab * np.log(p + eps) + (1 - lab) * np.log(1 - p + eps)) + np.sum(a ** 2) / (2 * tau ** 2) + 0.5 * (b - 0.7) ** 2 / 0.5 ** 2
    th0 = np.concatenate([[-12.0, 0.7], np.zeros(len(apts))]); r = minimize(nll, th0, method="L-BFGS-B")
    a0, b = r.x[0], r.x[1]; return {a_: (float(a0 + r.x[2 + i]), float(b)) for i, a_ in enumerate(apts)}

def dayshift_caps(trn_mask, unm, adep, ms, y):
    """Per-airport Laplace-smoothed P(day-shift | unmatched, MVT-SCHED > 16 h) from TRAINING rows only. Used as the upper cap on q so the
    logistic cannot be more confident than the evidence behind it. No hand-set constant, no leaderboard input."""
    caps = {}
    for a_ in np.unique(adep):
        c_ = trn_mask & unm & (adep == a_) & (ms > 57600)
        n_ = int(c_.sum()); k_ = int((y[c_] > 80000).sum())
        caps[a_] = (k_ + 1.0) / (n_ + 2.0) if n_ > 0 else 0.5
    return caps

def dayshift_q(models, adep_v, ms_v):
    q = np.zeros(len(adep_v))
    for a_, (i_, b_) in models.items():
        m_ = adep_v == a_; q[m_] = 1.0 / (1.0 + np.exp(-(i_ + b_ * ms_v[m_] / 3600.0)))
    return q

def main():
    ap = argparse.ArgumentParser()
    import xsave_lib as _xs; _xs.add_args(ap)   # SAVE_MODEL
    ap.add_argument("--name", required=True); ap.add_argument("--feats", default="base,tl,ref,turn,te")
    ap.add_argument("--strategy", choices=["keep", "cleanpru", "mixture"], default="keep")
    ap.add_argument("--fb-tol", type=int, default=60, help="artefact definition |BLOCK-SCHED|<=tol s")
    ap.add_argument("--fb-min-delay", type=int, default=0, help="artefact only if MVT-SCHED > this (s); excludes genuinely punctual flights from the artefact class")
    ap.add_argument("--cv", default="janjul", help="janjul | forward | months:M1,M2 (hold out these months, equal weights)"); ap.add_argument("--model", choices=["lgbm", "cat", "xgb"], default="lgbm")
    ap.add_argument("--target", choices=["raw", "resid_aobt3", "resid_ref", "resid_clock", "mov", "rep"], default="raw")   # dark_distill_note: mov = predict the clock anchor MVT-leave (movement part), rep = predict y - anchor (reporting part), both with offset 0
    ap.add_argument("--fit-linked", action="store_true", help="dark_distill_note: fit the regressor/classifier on ADS-B-linked matched rows with a real clock (xlk_src <= 2) only; validation/OOF rows unchanged (all matched rows)")
    ap.add_argument("--linked-only", action="store_true", help="lgb_adsb_linked: train and validate on ADS-B-linked (adsb_linked == 1) matched rows only")
    ap.add_argument("--clk-lo", type=float, default=0.0); ap.add_argument("--clk-hi", type=float, default=7200.0)
    ap.add_argument("--only-heavy", action="store_true", help="APT_MEMBER2: train and validate on matched rows with heavy == 1 only (all airports)")
    ap.add_argument("--only-apt", default="", help="X2: train and validate on matched rows of these airports only (comma list)")
    ap.add_argument("--apt-weight", default="", help="dark_distill_note: soft dedication A[+B]:w — regressor sample weight 1 on rows of these airports, w elsewhere (all airports stay in the fit)")
    ap.add_argument("--unlinked-only", action="store_true", help="X2/dark_form: train and validate on NOT ADS-B-linked (adsb_linked != 1) matched rows only")
    ap.add_argument("--no-xlk-feats", action="store_true", help="lgb_adsb_linked: do not add xlk_src/xlk_anchor/xlk_a3d as features")
    ap.add_argument("--objective", choices=["l2", "huber"], default="l2"); ap.add_argument("--huber-delta", type=float, default=600)
    ap.add_argument("--per-airport", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="cuda: LightGBM CUDA tree learner (needs the CUDA build; categoricals are integer-coded and bagging is off because the CUDA learner crashes with native categoricals)")
    ap.add_argument("--lr", type=float, default=0.021); ap.add_argument("--leaves", type=int, default=423); ap.add_argument("--min-leaf", type=int, default=41)
    ap.add_argument("--l2", type=float, default=1.2); ap.add_argument("--ff", type=float, default=0.46); ap.add_argument("--rounds", type=int, default=6000); ap.add_argument("--fixed-rounds", type=int, default=0, help="train the regressor for exactly N rounds (no early stopping); 0 = early stopping"); ap.add_argument("--es-patience", type=int, default=150)
    ap.add_argument("--cat-depth", type=int, default=8); ap.add_argument("--cat-lr", type=float, default=0.06); ap.add_argument("--cat-iters", type=int, default=6000)
    ap.add_argument("--cls-rounds", type=int, default=800); ap.add_argument("--cls-leaves", type=int, default=127)
    ap.add_argument("--p-power", type=float, default=1.0, help="mixture: use p**power")
    ap.add_argument("--cls-weight", choices=["none", "stake", "sqrtstake", "mix", "direct", "dmix"], default="none", help="classifier sample weight: stake = (max(MVT-SCHED,0)/3600)^2 + 0.05 (what a misclassification costs), sqrtstake = its square root")
    ap.add_argument("--direct-huber", type=float, default=0.0, help="direct objective: Huber delta in hours on the mixture error (0 = squared error)")
    ap.add_argument("--direct-exclude-apt", default="", help="direct objective: airports excluded from its training rows (comma list)")
    ap.add_argument("--direct-rounds", type=int, default=400)
    ap.add_argument("--direct-lambda", type=float, default=1.0, help="direct objective: weight of the task gradient; (1-lambda) goes to stake-weighted log-loss gradient")
    ap.add_argument("--stake-cap-q", type=float, default=0, help="cap stake weights at this quantile (0 = no cap)")
    ap.add_argument("--pfeat", action="store_true", help="add out-of-fold artefact logit as a regressor feature")
    ap.add_argument("--gate-gamma", type=float, default=1.0, help="gate shrinkage: p~ = p^g / (p^g + (1-p)^g)")
    ap.add_argument("--drop-lift", type=float, default=0, help="only drop near-schedule rows at airports where the TRAINING data shows a lift of at least this factor in P(taxi > 1 h) for those rows vs the rest (0 = drop everywhere, current behaviour). Learned per airport, no literal airport names")
    ap.add_argument("--drop-tol", type=float, default=-1, help="regressor drops rows with |BLOCK-SCHED| <= this (seconds); -1 = same as --fb-tol. Decouples the regressor training mask from the classifier target, which wants a looser threshold")
    ap.add_argument("--cls-tail", default="", help="second classifier for TAIL rows only (unmatched & MVT-SCHED > --cls-tail-ms): comma list lr,leaves,min_leaf,ff,l2,cat_smooth,rounds,stake_power,p_power,stake_floor"); ap.add_argument("--cls-tail-ms", type=float, default=3600)
    ap.add_argument("--cls-lr", type=float, default=0.05); ap.add_argument("--cls-min-leaf", type=int, default=50); ap.add_argument("--cls-l2", type=float, default=0.0); ap.add_argument("--cls-cat-smooth", type=float, default=10.0); ap.add_argument("--stake-floor", type=float, default=0.05)
    ap.add_argument("--stake-power", type=float, default=2.0, help="classifier stake weight = (max(delay,0)/3600)^power + 0.05")
    ap.add_argument("--cls-ff", type=float, default=0.8)
    ap.add_argument("--cls-delayed-only", action="store_true", help="train the classifier only on rows with MVT-SCHED > 30 min (others get p=0)")
    ap.add_argument("--mix-drop", choices=["fbpru", "fb", "fbpru_unm", "fbpru_m"], default="fbpru", help="mixture: rows dropped from the regressor fit (fbpru = artefacts + taxi>2h, fb = artefacts only)")
    ap.add_argument("--trim-q", type=float, default=0, help="drop training rows above this within-airport target quantile (0=off)")
    ap.add_argument("--dayshift-midnight", action="store_true", help="day-shift share q: fit on unmatched departures that crossed a local midnight between SCHED and MVT and set q = 0 for non-crossers (2025: 0/10 non-crossers vs 12/40 crossers rolled over, all at LIRF)")
    ap.add_argument("--linear-tree", action="store_true", help="LightGBM linear_tree=True for the regressor (piecewise-linear leaves: extrapolates beyond the 2025 range on storm days)")
    ap.add_argument("--cat-gpu-ram", type=float, default=0.0, help="CatBoost gpu_ram_part (0 = library default 0.95); 0.3 keeps the GPU job at ~30 GB so LightGBM seeds can run beside it")
    ap.add_argument("--unm-masked", type=float, default=0.0, help="blend weight (0..1) of a regressor for UNMATCHED rows that is trained on MATCHED rows with every NM-derived column masked to missing (training distribution = the unmatched rows' feature pattern), offset 0 on both sides")
    ap.add_argument("--unm-masked-thr", type=float, default=0.98, help="a column counts as NM-derived when its null rate on unmatched training rows is >= this and < 0.5 on matched rows")
    ap.add_argument("--unm-reg", action="store_true", help="fit a separate taxi regressor on unmatched (no flight plan) rows and use it for those rows")
    ap.add_argument("--season-weight", type=float, default=1.0, help="regressor sample weight for training rows in the months adjacent (±1) to the target months (fold months, or Jan/Jul for ranking)")
    ap.add_argument("--finetune-rounds", type=int, default=0, help="after the main fit, continue boosting this many rounds on seasonal rows only (LightGBM)")
    ap.add_argument("--es-clean", action="store_true", help="early-stop the regressor on validation rows without substituted/>2h labels")
    ap.add_argument("--predict-rank", action="store_true"); ap.add_argument("--no-es", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments/results_v2.jsonl")); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mix-additive-mode", choices=["prop", "qfirst"], default="qfirst", help="how to reconcile p+q > 1 in the additive mixture. 'prop' scales both shares down; 'qfirst' keeps q (fitted directly on day-shift labels) and gives p the remainder, which coincides with the nested form as p -> 1 and differs only where p is low -- the case the nested form gets wrong. Measured on the ranking rows: prop +0.84 s of expected board, qfirst -0.28 s")
    ap.add_argument("--mix-additive", action="store_true", help="treat substitution and day-shift as the near-disjoint mechanisms they are: pred = p*(MVT-SCHED) + q*(86400+t_ds) + (1-p-q)*taxi, instead of nesting the day-shift value inside the artefact branch (where its mass is p*q and the fb classifier -- which correctly gives a day-shifted row a LOW p, since its BLOCK is a day out rather than within 60 s of SCHED -- suppresses the branch meant to catch it)")
    ap.add_argument("--tail-gate-lobt", action="store_true", help="widen the tail gate with |AOBT_3-LOBT| > 2400 s, fit the gate regressor on matched & not-fb rows WITHOUT the y cap, and use MVT-LOBT as its offset (stale-AOBT rows follow LOBT: diagnostic probe, matched rows with AOBT_3-LOBT > 5000 have median y-(MVT-AOBT_3) = +5096 but y-(MVT-LOBT) = -1066)")
    ap.add_argument("--tail-gate", type=float, default=0.0, help="blend weight of a second regressor fitted only on matched rows with an abnormal NM offset (MVT-AOBT_3 < 300 or > 2200, or MVT-EOBT_1 > 3600) and applied only inside that gate (~3 %% of rows, ~12 %% of squared error per a competitor journal). 0 = off")
    ap.add_argument("--train-months", default="", help="comma list of months to TRAIN on (others neither trained nor validated, except the --cv validation months). Temporal-extrapolation folds: --train-months 1,2,3,4,5,6 --cv months:12 trains on Jan-Jun and validates December with a six-month gap, the analogue of predicting July 2026 from data ending in December 2025")
    ap.add_argument("--drop-cols", default="", help="comma list of column-name prefixes to exclude from the feature list after group resolution (e.g. te_fno_,atfm_allpre). Used to remove features whose 2026 missingness pattern never occurs in training -- nan_shift.py")
    ap.add_argument("--ylog", action="store_true", help="regressor fits log1p(y) on the raw label instead of the residual over the offset (the offset stays a feature); predictions are expm1 with a Duan smearing ratio fitted on a training subsample. Bounded influence of huge labels; Motoki blends this parametrisation with the linear one")
    ap.add_argument("--bagging", type=float, default=0.8, help="LightGBM bagging_fraction for the regressor (freq 1); 0.8 has been the hard-coded value all along")
    ap.add_argument("--reg-pweight", type=float, default=0.0, help="XA/E6: regressor sample weight (1-p)^K with p an out-of-fold (month-grouped, 4-fold cross-fitted) classifier probability on training rows; the mixed prediction weights the taxi branch by (1-p), so its squared-error gradient carries (1-p)^2. 0 = off (default)")
    ap.add_argument("--reg-pclip", type=float, default=0.9, help="XA/E6: clip the cross-fitted p at this value before weighting / target construction ((1-p) -> 0 instability)")
    ap.add_argument("--reg-mixtarget", action="store_true", help="XA/E6b: fit the regressor on ALL training rows (fb included; only non-fb >2 h / day-shift rows dropped) with target t = (y - p*art)/(1-p) and weight (1-p)^2, i.e. the exact gradient of the mixed loss (p*art + (1-p)*taxi - y)^2 for fixed cross-fitted p and art = MVT-SCHED")
    ap.add_argument("--mixtarget-clip", type=float, default=3600.0, help="XA/E6b: clip |t - y| at this many seconds")
    ap.add_argument("--reuse-taxi", default="", help="XA: fold path only -- skip the regressors and take the taxi branch (row 3) of data/oof/<PREFIX>_<fold>*.npy for the validation rows, so classifier variants are compared on an identical taxi branch")
    ap.add_argument("--direct-proxy-rounds", type=int, default=300, help="direct objective: rounds of the cross-fitted proxy taxi regressor (300 = the 6 Sep value)")
    ap.add_argument("--rank-p-only", action="store_true", help="XA: production classifier only -- fit the classifier (with --cls-weight etc.) on all labelled rows, write data/preds/<name>.parquet with MVT_ID_mvt, p_raw for the ranking rows, and exit (no regressor, no fold)")
    ap.add_argument("--l2-refit-rounds", type=int, default=0, help="XA/H3: after a robust (--objective huber) regressor, continue boosting this many rounds with the L2 objective from its predictions (re-centres the leaves on the conditional mean). 0 = off")
    ap.add_argument("--f32", action="store_true", help="build the feature matrix in float32 instead of float64 (halves peak memory; LightGBM bins the values anyway so the fitted model is unchanged in practice)")
    ap.add_argument("--features", default=str(ROOT / "data/features_v2.parquet")); ap.add_argument("--dayshift-cap", type=float, default=-1.0, help="upper cap on the day-shift share q. -1 = AUTO: per-airport Laplace-smoothed rate (k+1)/(n+2) of day-shifted labels among that airport's unmatched training departures with MVT-SCHED > 16 h — derived from training data only, no hand-set constant. A value >= 0 sets a fixed cap"); ap.add_argument("--dayshift-pooled", type=float, default=0.0, help="use the hierarchical (partially pooled) day-shift model with this intercept-prior sd tau (0 = per-airport independent logistic)"); ap.add_argument("--dayshift", action="store_true", help="third mixture component: for unmatched LIRF rows with MVT-SCHED > 16h, part of the artefact mass is 86400 s + taxi (day-shifted block time); share estimated on the training rows")
    a = ap.parse_args(); t0 = time.time(); _xs.setup(a, __file__)   # SAVE_MODEL
    # load only the columns this run needs (feature groups are resolved from the schema first) -- keeps memory at ~half of the full frame
    _schema = list(pl.read_parquet_schema(a.features).keys()); _dummy = pl.DataFrame(schema={c: pl.Null for c in _schema})
    _feats = list(CAT_BASE)
    for g in a.feats.split(","): _feats += cols_for(_dummy, g)
    _meta = ["MVT_ID_mvt", "y", "m", "is_rank", "ADEP_mvt", "fb", "fb_pru", "mvt_sched", "mvt_aobt3", "mvt_eobt1", "mvt_lobt", "ref_sr_p10", "unmatched", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"] + (["heavy"] if a.only_heavy else [])
    d = pl.read_parquet(a.features, columns=[c for c in dict.fromkeys(_meta + _feats) if c in _schema])
    # side tables: feature groups kept in their own parquet (keyed by MVT_ID_mvt) so a new group does not need a
    # full feature rebuild. Joined only when the group is requested.
    for _g, _f in (("storm", ROOT / "data/external/storm_features.parquet"), ("reg", ROOT / "data/external/reg_features.parquet"), ("dur", ROOT / "data/external/dur_features.parquet"), ("sarr", ROOT / "data/external/sarr_features.parquet"), ("durp", ROOT / "data/external/durp_features.parquet"), ("fbte", ROOT / "data/external/fbte_features.parquet"), ("dreg", ROOT / "data/external/dreg_features.parquet"), ("dreg2", ROOT / "data/external/dreg2_features.parquet"), ("areg", ROOT / "data/external/areg_features.parquet"), ("rreg", ROOT / "data/external/rreg_features.parquet"), ("dreg3", ROOT / "data/external/dreg3_features.parquet"), ("dreg3perm", ROOT / "data/external/dreg3perm_features.parquet"), ("preg", ROOT / "data/external/preg_features.parquet"), ("wxs", ROOT / "data/external/wxs_features.parquet"), ("nbrs", ROOT / "data/external/nbrs_features.parquet"), ("nbrsc", ROOT / "data/external/nbrsc_features.parquet"), ("adsb", ROOT / "data/external/adsb_features.parquet"), ("adsb2", ROOT / "data/external/adsb2_features.parquet"), ("adsbF", ROOT / "data/external/adsbF_features.parquet"), ("adsb3", ROOT / "data/external/adsb3_features.parquet"), ("adsb4", ROOT / "data/external/adsb4_features.parquet"), ("adsbU", ROOT / "data/external/adsbU_features.parquet"), ("arrt", ROOT / "data/external/arrt_features.parquet"), ("xe", ROOT / "data/external/xe_queue_features.parquet"), ("xb39", ROOT / "data/external/xb_m39_features.parquet"), ("xb28", ROOT / "data/external/xb_m28_features.parquet"), ("xb410", ROOT / "data/external/xb_m410_features.parquet"), ("xb17", ROOT / "data/external/xb_m17_features.parquet"), ("xbP", ROOT / "data/external/xb_prod_features.parquet"), ("xba", ROOT / "data/external/xba_features.parquet")):
        if _g in a.feats.split(","):
            if not _f.exists(): raise SystemExit(f"side table for group {_g} missing: {_f}")
            d = d.join(pl.read_parquet(_f), on="MVT_ID_mvt", how="left")
    if a.fb_tol != 60 or a.fb_min_delay > 0:
        d = d.with_columns((((pl.col("BLOCK_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= a.fb_tol) & (pl.col("mvt_sched") > a.fb_min_delay)).cast(pl.Int8).alias("fb"))
        d = d.with_columns(((pl.col("fb") == 1) | (pl.col("y") > 7200)).cast(pl.Int8).alias("fb_pru"))
    _need_clk = a.target in ("resid_clock", "mov", "rep") or a.linked_only or a.unlinked_only or bool(a.only_apt) or a.fit_linked
    if _need_clk:
        for _c in ("adsb_linked", "adsb2_leave", "adsb3_mv_gs", "adsb_move"):
            if _c not in d.columns: raise SystemExit(f"lgb_adsb_linked: column {_c} missing -- add feats adsb,adsb2,adsb3")
        def _ok(c): return pl.col(c).is_not_null() & pl.col(c).is_not_nan() & (pl.col(c) >= a.clk_lo) & (pl.col(c) <= a.clk_hi)
        d = d.with_columns(xlk_src=pl.when(_ok("adsb2_leave")).then(0).when(_ok("adsb3_mv_gs")).then(1).when(_ok("adsb_move")).then(2).when(pl.col("mvt_aobt3").is_not_null()).then(3).otherwise(4).cast(pl.Float64))
        d = d.with_columns(xlk_anchor=pl.when(pl.col("xlk_src") == 0).then(pl.col("adsb2_leave")).when(pl.col("xlk_src") == 1).then(pl.col("adsb3_mv_gs")).when(pl.col("xlk_src") == 2).then(pl.col("adsb_move")).when(pl.col("xlk_src") == 3).then(pl.col("mvt_aobt3")).otherwise(None).cast(pl.Float64))
        d = d.with_columns(xlk_a3d=(pl.col("mvt_aobt3") - pl.col("xlk_anchor")).cast(pl.Float64), xlk_lk=pl.col("adsb_linked").fill_null(0).cast(pl.Int8))
    groups = a.feats.split(","); feats = list(CAT_BASE)
    for g in groups: feats += cols_for(d, g)
    feats = [f for f in feats if f in d.columns]
    if _need_clk and not a.no_xlk_feats: feats += ["xlk_src", "xlk_anchor", "xlk_a3d"]
    if a.drop_cols:
        _dp = tuple(x for x in a.drop_cols.split(",") if x); feats = [f for f in feats if not f.startswith(_dp)]
    cats = [c for c in CAT_BASE if c in feats]
    if a.model == "lgbm" and a.device != "cuda":
        # single float64 block built column-chunk-wise (categoricals as codes, -1 -> NaN); numerically identical for LightGBM since
        # categorical_feature is passed by name, and avoids ~3 full-size copies of the frame (peak memory ~23 GB -> ~14 GB)
        import pandas as pd
        _dt = np.float32 if a.f32 else np.float64
        Xa = np.empty((d.height, len(feats)), dtype=_dt)
        for j0 in range(0, len(feats), 16):
            chunk = feats[j0:j0 + 16]; pdf = d.select(chunk).to_pandas()
            for j, c in enumerate(chunk):
                if c in cats:
                    codes = (_xs.cat(pdf[c], c) if _xs.ON else pdf[c].astype("category")).cat.codes.to_numpy().astype(_dt); codes[codes < 0] = np.nan; Xa[:, j0 + j] = codes
                else: Xa[:, j0 + j] = pdf[c].to_numpy(dtype=np.float64, na_value=np.nan).astype(_dt, copy=False)
            del pdf
        X = pd.DataFrame(Xa, columns=feats); del Xa
    else:
        X = d.select(feats).to_pandas()
        for c in cats: X[c] = X[c].astype("category")
    if a.device == "cuda" and a.model == "lgbm":
        for c in cats: X[c] = X[c].cat.codes.astype("float32")
        cats = []
    if a.pfeat:
        # out-of-fold artefact logit (stake-weighted log-loss classifier) as an extra regressor input; test rows get the full-train model
        _fb = d["fb"].to_numpy() == 1; _ms = d["mvt_sched"].to_numpy().astype(float); _m = d["m"].to_numpy(); _rank = (d["is_rank"] == 1).to_numpy()
        _st = (np.maximum(_ms, 0) / 3600.0) ** a.stake_power + a.stake_floor
        if a.stake_cap_q > 0: _st = np.minimum(_st, np.quantile(_st, a.stake_cap_q))
        _prm = dict(objective="binary", learning_rate=0.05, num_leaves=a.cls_leaves, min_data_in_leaf=50, feature_fraction=a.cls_ff, verbose=-1, num_threads=a.threads if a.threads else 0, cat_smooth=10, seed=a.seed, **({"device_type": "cuda"} if (a.device == "cuda" and a.model == "lgbm") else {}))
        _z = np.zeros(len(_fb)); _tr = np.where(~_rank)[0]; _rng = np.random.default_rng(a.seed); _fold = _rng.integers(0, 4, len(_tr))
        for k in range(4):
            fit_i = _tr[_fold != k]; pr_i = _tr[_fold == k]
            _b = lgb.train(_prm, lgb.Dataset(X.iloc[fit_i], _fb[fit_i].astype(int), weight=_st[fit_i], categorical_feature=cats), a.cls_rounds)
            _z[pr_i] = _b.predict(X.iloc[pr_i], raw_score=True)
        _b = lgb.train(_prm, lgb.Dataset(X.iloc[_tr], _fb[_tr].astype(int), weight=_st[_tr], categorical_feature=cats), a.cls_rounds)
        _z[np.where(_rank)[0]] = _b.predict(X.iloc[np.where(_rank)[0]], raw_score=True)
        X["p_art_logit"] = _z; feats = feats + ["p_art_logit"]
    y = d["y"].to_numpy(); m = d["m"].to_numpy(); is_rank = (d["is_rank"] == 1).to_numpy(); adep = d["ADEP_mvt"].to_numpy()
    fb = d["fb"].to_numpy() == 1; fbp = d["fb_pru"].to_numpy() == 1; ms = d["mvt_sched"].to_numpy().astype(float)
    aobt = np.nan_to_num(d["mvt_aobt3"].to_numpy().astype(float), nan=0.0); ref = np.nan_to_num(d["ref_sr_p10"].to_numpy().astype(float), nan=0.0)
    unm_ = d["unmatched"].to_numpy() == 1; mvt_id = d["MVT_ID_mvt"].to_numpy()
    hv_ = (d["heavy"].fill_null(0).to_numpy() == 1) if a.only_heavy else None   # APT_MEMBER2
    if _need_clk:
        anchor = np.nan_to_num(d["xlk_anchor"].to_numpy().astype(float), nan=0.0); xlk_src_v = d["xlk_src"].to_numpy().astype(int); lk_ = d["xlk_lk"].to_numpy() == 1
    aobt_present = d["mvt_aobt3"].is_not_null().to_numpy(); d_eobt1 = d["mvt_eobt1"].to_numpy().astype(float) if "mvt_eobt1" in d.columns else np.zeros(len(y))
    d_lobt = d["mvt_lobt"].to_numpy().astype(float) if "mvt_lobt" in d.columns else np.zeros(len(y))
    bs_ = (d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds().abs().to_numpy()
    # crossed a local midnight between SCHED and MVT (crude local time: UTC + winter offset, +1 h Apr-Oct except LTFM) — for --dayshift-midnight
    _tz = {"LIRF": 1, "EHAM": 1, "EDDF": 1, "EDDM": 1, "LFPG": 1, "LSZH": 1, "LEBL": 1, "LEMD": 1, "EGLL": 0, "LTFM": 3}
    _dl = d.select(pl.col("ADEP_mvt").replace_strict(_tz, default=1).alias("tz"), "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
    def _loc(c): return pl.col(c) + pl.duration(hours=pl.col("tz") + pl.when((pl.col(c).dt.month() >= 4) & (pl.col(c).dt.month() <= 10) & (pl.col("tz") != 3)).then(1).otherwise(0))
    cross_ = ((_loc("MVT_TIME_UTC_mvt").dt.date() - _loc("SCHED_TIME_UTC_mvt").dt.date()).dt.total_days() >= 1).fill_null(False)
    cross_ = _dl.select(cross_.alias("c"))["c"].to_numpy().astype(bool); del _dl   # |block - schedule|, needed by --drop-tol/--drop-lift after the frame is freed
    import gc; del d; gc.collect()   # frame no longer needed
    if a.cv == "janjul":
        val = (~is_rank) & np.isin(m, [1, 7]); trn = (~is_rank) & ~np.isin(m, [1, 7])
        w = np.zeros(len(y)); w[val & (m == 1)] = 0.44 / (val & (m == 1)).sum(); w[val & (m == 7)] = 0.56 / (val & (m == 7)).sum()
    elif a.cv == "forward":
        val = (~is_rank) & np.isin(m, [11, 12]); trn = (~is_rank) & (m <= 10); w = np.zeros(len(y)); w[val] = 1.0 / val.sum()
    else:
        hold = [int(x) for x in a.cv.split(":")[1].split(",")]
        val = (~is_rank) & np.isin(m, hold); trn = (~is_rank) & ~np.isin(m, hold); w = np.zeros(len(y)); w[val] = 1.0 / val.sum()
    if a.train_months:
        # temporal-extrapolation fold: train only on the listed months (validation months unchanged)
        _tm = [int(x) for x in a.train_months.split(",") if x]; trn = trn & np.isin(m, _tm)
    if a.linked_only or a.unlinked_only or a.only_apt or a.only_heavy:
        _sel = (lk_ if a.linked_only else (~lk_ if a.unlinked_only else np.ones(len(y), bool))) & ~unm_
        if a.only_apt: _sel = _sel & np.isin(adep, a.only_apt.split(","))
        if a.only_heavy: _sel = _sel & hv_   # APT_MEMBER2
        trn = trn & _sel; val = val & _sel
        if a.cv == "janjul":
            w = np.zeros(len(y)); w[val & (m == 1)] = 0.44 / max((val & (m == 1)).sum(), 1); w[val & (m == 7)] = 0.56 / max((val & (m == 7)).sum(), 1)
        else: w = np.zeros(len(y)); w[val] = 1.0 / val.sum()
    if a.fit_linked: trn = trn & lk_ & ~unm_ & (xlk_src_v <= 2)   # dark_distill_note: distillation — fit on linked rows with a real clock, predict everywhere
    off = {"raw": np.zeros(len(y)), "resid_aobt3": aobt, "resid_ref": ref, "resid_clock": (anchor if a.target == "resid_clock" else None), "mov": np.zeros(len(y)), "rep": np.zeros(len(y))}[a.target]
    if _need_clk:
        res_xlk = dict(n_trn=int(trn.sum()), n_val=int(val.sum()), src_trn={int(k): int(((xlk_src_v == k) & trn).sum()) for k in range(5)}, src_val={int(k): int(((xlk_src_v == k) & val).sum()) for k in range(5)}, src_rank_linked={int(k): int(((xlk_src_v == k) & is_rank & lk_ & ~unm_).sum()) for k in range(5)})
        print("lgb_adsb_linked", json.dumps(res_xlk), flush=True)
    drop = np.zeros(len(y), bool)
    if a.strategy == "cleanpru" or (a.strategy == "mixture" and a.mix_drop == "fbpru"): drop |= fbp
    if a.strategy == "mixture" and a.mix_drop == "fbpru_unm": drop |= fb | (unm_ & (y > 7200))
    if a.strategy == "mixture" and a.mix_drop == "fbpru_m": drop |= fb | ((~unm_) & (y > 7200)) | (y > 80000)   # reviewer: keep unmatched 2 h-22 h rows in the fit (unm_LIRF class), drop them on matched rows and the day-shift rows everywhere
    if a.strategy == "mixture" and a.mix_drop == "fb":
        _near = fb if a.drop_tol < 0 else (bs_ <= a.drop_tol)
        if a.drop_lift > 0:
            # keep the near-schedule rows in regressor training except at airports where they really are artefact-prone (learned on train rows)
            keep_apt = set(); res_lift = {}
            for a_ in np.unique(adep):
                m_ = trn & (adep == a_); n_ = m_ & _near; o_ = m_ & ~_near
                if n_.sum() >= 100 and o_.sum() >= 100:
                    p1 = float((y[n_] > 3600).mean()); p0 = float((y[o_] > 3600).mean())
                    lift = p1 / p0 if p0 > 0 else (np.inf if p1 > 0 else 0.0); res_lift[a_] = round(lift, 1)
                    if lift >= a.drop_lift: keep_apt.add(a_)
            drop |= (_near & np.isin(adep, list(keep_apt)))
            res_droplift = dict(lift=res_lift, dropped_at=sorted(keep_apt), n_dropped=int((_near & np.isin(adep, list(keep_apt))).sum()), n_near=int(_near.sum()))
        else: drop |= _near
    if a.trim_q > 0:
        for ap_ in np.unique(adep):
            mk = (adep == ap_) & trn; q = np.quantile(y[mk], a.trim_q); drop |= mk & (y > q)
    if a.reg_mixtarget and a.strategy == "mixture": drop = fbp & ~fb   # keep substituted rows; still drop >2 h / day-shift non-fb rows
    fit_mask = trn & ~drop
    regw = None; ymod = y   # XA: regressor weights / modified target (set below when --reg-pweight / --reg-mixtarget)
    if a.target == "mov": ymod = anchor.copy()   # dark_distill_note: movement part MVT - leave (anchor) as the regressor target
    if a.target == "rep": ymod = y - anchor         # dark_distill_note: reporting part leave - BLOCK
    if a.apt_weight:   # dark_distill_note: soft dedication (regressor only; the classifier is unchanged)
        _aw, _ww = a.apt_weight.split(":"); regw = np.where(np.isin(adep, _aw.split("+")), 1.0, float(_ww))
    thr = a.threads if a.threads else 0

    def season_mask(target_months):
        adj = set()
        for mm in target_months:
            for k in (-1, 0, 1): adj.add(((mm - 1 + k) % 12) + 1)
        return np.isin(m, list(adj))
    def fit_reg(fi, vi, pi, es=True, rounds=None, target_months=None):
        ytr = np.log1p(np.maximum(y, 0.0)) if a.ylog else ymod - off
        if target_months is None:
            target_months = [1, 7] if a.cv == "janjul" else ([11, 12] if a.cv == "forward" else [int(x) for x in a.cv.split(":")[1].split(",")])
        smask = season_mask(target_months)
        if a.model == "lgbm":
            obj = dict(objective="regression") if a.objective == "l2" else dict(objective="huber", alpha=a.huber_delta)
            prm = dict(learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=a.ff, bagging_fraction=a.bagging, bagging_freq=1,
                       lambda_l2=a.l2, verbose=-1, num_threads=thr, cat_l2=10, cat_smooth=20, seed=a.seed, **obj)
            if a.device == "cuda": prm.update(device_type="cuda", bagging_freq=0, bagging_fraction=1.0)
            if a.linear_tree: prm["linear_tree"] = True
            wtr = np.where(smask[fi], a.season_weight, 1.0) if a.season_weight != 1.0 else None
            if regw is not None: wtr = regw[fi] if wtr is None else wtr * regw[fi]
            ds = lgb.Dataset(X.iloc[fi], ytr[fi], weight=wtr, categorical_feature=cats)
            if es and a.fixed_rounds == 0:
                ve = vi[~fbp[vi]] if a.es_clean else vi
                dv = lgb.Dataset(X.iloc[ve], ytr[ve], weight=w[ve], reference=ds)
                b = lgb.train(prm, ds, a.rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(a.es_patience, verbose=False)]); it = b.best_iteration
            else:
                n_ = a.fixed_rounds if (es and a.fixed_rounds) else rounds
                b = lgb.train(prm, ds, n_); it = n_
            pred_ = b.predict(X.iloc[pi], num_iteration=it)
            if a.ylog:
                # back-transform with a smearing ratio so the mean is unbiased for an RMSE metric
                sub = fi if len(fi) <= 200000 else np.random.default_rng(a.seed).choice(fi, 200000, replace=False)
                smear = float(np.mean(y[sub]) / max(np.mean(np.expm1(b.predict(X.iloc[sub], num_iteration=it))), 1e-9))
                return np.expm1(pred_) * smear, it
            if a.l2_refit_rounds > 0 and a.objective != "l2":
                init_ = b.predict(X.iloc[fi], num_iteration=it)
                br = lgb.train(dict(prm, objective="regression"), lgb.Dataset(X.iloc[fi], ytr[fi], weight=wtr, init_score=init_, categorical_feature=cats), a.l2_refit_rounds)
                pred_ = pred_ + br.predict(X.iloc[pi], raw_score=True)
            if a.finetune_rounds > 0:
                fs = fi[smask[fi]]
                if len(fs) > 1000:
                    init = b.predict(X.iloc[fs], num_iteration=it)
                    prm_ft = dict(prm, learning_rate=prm["learning_rate"] * 0.5)
                    bf = lgb.train(prm_ft, lgb.Dataset(X.iloc[fs], ytr[fs], init_score=init, categorical_feature=cats), a.finetune_rounds)
                    pred_ = pred_ + bf.predict(X.iloc[pi], raw_score=True)
            return pred_ + off[pi], it
        elif a.model == "xgb":
            import xgboost as xgb
            gpu = os.environ.get("CAT_GPU", "0") == "1"
            prm = dict(objective="reg:squarederror", learning_rate=a.lr, max_depth=0, max_leaves=a.leaves, min_child_weight=a.min_leaf, reg_lambda=a.l2,
                       colsample_bytree=a.ff, subsample=0.8, tree_method="hist", grow_policy="lossguide", max_bin=255, max_cat_to_onehot=8,
                       device="cuda" if (gpu or a.device == "cuda") else "cpu", nthread=thr if thr else os.cpu_count(), seed=a.seed)
            dtr = xgb.DMatrix(X.iloc[fi], ytr[fi], enable_categorical=True)
            if es and a.fixed_rounds:
                b = xgb.train(prm, dtr, a.fixed_rounds); it = a.fixed_rounds
            elif es:
                ve = vi[~fbp[vi]] if a.es_clean else vi
                dv = xgb.DMatrix(X.iloc[ve], ytr[ve], weight=w[ve], enable_categorical=True)
                b = xgb.train(prm, dtr, a.rounds, evals=[(dv, "val")], early_stopping_rounds=150, verbose_eval=False); it = b.best_iteration + 1
            else:
                b = xgb.train(prm, dtr, rounds); it = rounds
            return b.predict(xgb.DMatrix(X.iloc[pi], enable_categorical=True), iteration_range=(0, it)) + off[pi], it
        else:
            from catboost import CatBoostRegressor, Pool
            gpu = os.environ.get("CAT_GPU", "0") == "1"
            Xc = X.copy()
            for c in cats: Xc[c] = Xc[c].astype(str).replace({"nan": "NA", "None": "NA"}).fillna("NA")
            n_cat = (a.fixed_rounds if (es and a.fixed_rounds) else (rounds or a.cat_iters))
            prm = dict(iterations=n_cat, learning_rate=a.cat_lr, depth=a.cat_depth, loss_function="RMSE", random_seed=a.seed, verbose=0,
                       task_type="GPU" if gpu else "CPU", thread_count=thr if thr else -1, l2_leaf_reg=3, one_hot_max_size=16)
            if gpu and a.cat_gpu_ram > 0: prm["gpu_ram_part"] = a.cat_gpu_ram   # GPU workstation unified memory: the default 0.95 reserves ~85 GB of the 119 GB for the whole run
            mdl = CatBoostRegressor(**prm)
            if es and a.fixed_rounds == 0:
                ve = vi[~fbp[vi]] if a.es_clean else vi
                mdl.fit(Pool(Xc.iloc[fi], ytr[fi], cat_features=cats, weight=(regw[fi] if regw is not None else None)), eval_set=Pool(Xc.iloc[ve], ytr[ve], cat_features=cats, weight=w[ve]), early_stopping_rounds=200, use_best_model=True)
                it = mdl.get_best_iteration()
            else:
                mdl.fit(Pool(Xc.iloc[fi], ytr[fi], cat_features=cats, weight=(regw[fi] if regw is not None else None))); it = n_cat
            return mdl.predict(Pool(Xc.iloc[pi], cat_features=cats)) + off[pi], it

    def fit_cls(fi, pi, ov=None):
        o = ov or {}
        prm = dict(objective="binary", learning_rate=o.get("lr", a.cls_lr), num_leaves=int(o.get("leaves", a.cls_leaves)), min_data_in_leaf=int(o.get("min_leaf", a.cls_min_leaf)), lambda_l2=o.get("l2", a.cls_l2), feature_fraction=o.get("ff", a.cls_ff), verbose=-1, num_threads=thr, cat_smooth=o.get("cat_smooth", a.cls_cat_smooth), seed=a.seed)
        if a.device == "cuda" and a.model == "lgbm": prm.update(device_type="cuda")
        if a.cls_delayed_only:
            fi = fi[ms[fi] > 1800]
        st = (np.maximum(ms[fi], 0) / 3600.0) ** o.get("stake_power", a.stake_power) + o.get("stake_floor", a.stake_floor)
        if a.stake_cap_q > 0:
            st = np.minimum(st, np.quantile(st, a.stake_cap_q))   # cap the heaviest stakes so a few huge delays do not dominate the classifier
        if a.cls_weight in ("direct", "dmix"):
            # Direct-RMSE objective (optionally Huber-robust, optionally excluding airports from its training rows);
            # dmix = mean of the stake-weighted log-loss probability and the direct probability.
            f_oof = np.zeros(len(y)); K = 4; rng = np.random.default_rng(a.seed); folds = rng.integers(0, K, len(fi))
            for k in range(K):
                tr_k = fi[folds != k]; te_k = fi[folds == k]; keep = tr_k[~fb[tr_k]]
                bk = lgb.train(dict(objective="regression", learning_rate=0.1, num_leaves=127, min_data_in_leaf=50, feature_fraction=0.8, verbose=-1, num_threads=thr, seed=a.seed),
                               lgb.Dataset(X.iloc[keep], (y - off)[keep], categorical_feature=cats), a.direct_proxy_rounds)
                f_oof[te_k] = bk.predict(X.iloc[te_k]) + off[te_k]
            excl = set(x for x in a.direct_exclude_apt.split(",") if x)
            m_fd = ~np.isin(adep[fi], list(excl)) if excl else np.ones(len(fi), bool)
            fd = fi[m_fd]; st_fd = st[m_fd]; y_fd = fb[fd].astype(float)
            gap_fd = (ms[fd] - f_oof[fd]) / 3600.0; res_fd = (f_oof[fd] - y[fd]) / 3600.0
            lam = float(a.direct_lambda); _scale = {}
            def fobj(preds, data):
                s_ = 1 / (1 + np.exp(-preds)); err = s_ * gap_fd + res_fd
                if a.direct_huber > 0:
                    dl = np.clip(err, -a.direct_huber, a.direct_huber)   # d/d err of Huber
                    grad = 2 * dl * gap_fd * s_ * (1 - s_); hess = 2 * (gap_fd * s_ * (1 - s_)) ** 2 * (np.abs(err) <= a.direct_huber) + 1e-4
                else:
                    grad = 2 * err * gap_fd * s_ * (1 - s_); hess = 2 * (gap_fd * s_ * (1 - s_)) ** 2 + 1e-4
                if lam < 1.0:
                    # lambda-interpolation: (1-lambda) of the stake-weighted log-loss gradient, rescaled once so both terms have the same mean magnitude
                    g_l = (s_ - y_fd) * st_fd; h_l = s_ * (1 - s_) * st_fd + 1e-6
                    if "c" not in _scale:
                        _scale["c"] = (np.mean(np.abs(grad)) + 1e-12) / (np.mean(np.abs(g_l)) + 1e-12)
                    c = _scale["c"]
                    grad = lam * grad + (1 - lam) * c * g_l; hess = lam * hess + (1 - lam) * c * h_l
                return np.clip(grad, -50, 50), hess
            warm = lgb.train(prm, lgb.Dataset(X.iloc[fi], fb[fi].astype(int), weight=st, categorical_feature=cats), a.cls_rounds)
            init = warm.predict(X.iloc[fd], raw_score=True)
            ds_c = lgb.Dataset(X.iloc[fd], fb[fd].astype(int), init_score=init, categorical_feature=cats)
            prm_d = dict(prm, objective=fobj, learning_rate=0.03, min_data_in_leaf=200, lambda_l2=10)
            bd = lgb.train(prm_d, ds_c, a.direct_rounds)
            z = warm.predict(X.iloc[pi], raw_score=True) + bd.predict(X.iloc[pi], raw_score=True)
            out = 1 / (1 + np.exp(-z))
            if a.cls_weight == "dmix":
                out = 0.5 * (out + warm.predict(X.iloc[pi]))
            if a.cls_delayed_only: out = np.where(ms[pi] > 1800, out, 0.0)
            return out
        if a.cls_weight == "mix":
            # ensemble of an unweighted classifier (sharpened with p_power) and a stake-weighted one (raw); p_power is then NOT applied again
            b0 = lgb.train(prm, lgb.Dataset(X.iloc[fi], fb[fi].astype(int), categorical_feature=cats), a.cls_rounds)
            b1 = lgb.train(prm, lgb.Dataset(X.iloc[fi], fb[fi].astype(int), weight=st, categorical_feature=cats), a.cls_rounds)
            out = 0.5 * (np.clip(b0.predict(X.iloc[pi]), 0, 1) ** a.p_power + b1.predict(X.iloc[pi]))
            a._mix_done = True
            if a.cls_delayed_only: out = np.where(ms[pi] > 1800, out, 0.0)
            return out
        wcls = None
        if a.cls_weight != "none":
            wcls = st if a.cls_weight == "stake" else np.sqrt(st)
        b = lgb.train(prm, lgb.Dataset(X.iloc[fi], fb[fi].astype(int), weight=wcls, categorical_feature=cats), int(o.get("rounds", a.cls_rounds)))
        out = b.predict(X.iloc[pi])
        if a.cls_delayed_only:
            out = np.where(ms[pi] > 1800, out, 0.0)
        return out

    def masked_unm_fit(fi_m, pi_u, rounds, trn_mask):
        """Regressor for UNMATCHED rows: fit on MATCHED rows (fi_m) whose NM-derived columns are masked to missing and whose
        `unmatched`/`apt_unm` flags are set to the unmatched pattern; offset 0 (raw taxi target). X and off are restored afterwards."""
        ui = np.where(trn_mask & unm)[0]; mi = np.where(trn_mask & ~unm)[0]
        rng_ = np.random.default_rng(0); ui_s = ui if len(ui) <= 200000 else rng_.choice(ui, 200000, replace=False); mi_s = rng_.choice(mi, min(200000, len(mi)), replace=False)
        nu = X.iloc[ui_s].isna().mean(); nm = X.iloc[mi_s].isna().mean()
        nmcols = [c for c in X.columns if c not in ("unmatched", "apt_unm") and float(nu[c]) >= a.unm_masked_thr and float(nm[c]) < 0.5]
        saved = {}
        for c in nmcols:
            j = X.columns.get_loc(c); saved[c] = X.iloc[fi_m, j].to_numpy(copy=True); X.iloc[fi_m, j] = np.nan
        if "unmatched" in X.columns:
            j = X.columns.get_loc("unmatched"); saved["unmatched"] = X.iloc[fi_m, j].to_numpy(copy=True); X.iloc[fi_m, j] = 1.0
        if "apt_unm" in X.columns:
            j = X.columns.get_loc("apt_unm"); saved["apt_unm"] = X.iloc[fi_m, j].to_numpy(copy=True)
            col = X.iloc[:, j].to_numpy(); newv = col[fi_m].copy()
            for g_ in np.unique(adep[fi_m]):
                src = ui[adep[ui] == g_]
                if len(src):
                    vals, cnt = np.unique(col[src][~pd.isna(col[src])], return_counts=True) if str(X.dtypes.iloc[j]) != "category" else (X.iloc[src, j].mode().to_numpy(), None)
                    if len(vals): newv[adep[fi_m] == g_] = vals[np.argmax(cnt)] if cnt is not None else vals[0]
            X.iloc[fi_m, j] = newv
        _off_save = off.copy(); off[:] = 0.0
        try:
            pu, _ = fit_reg(fi_m, None, pi_u, es=False, rounds=rounds)
        finally:
            off[:] = _off_save
            for c, v in saved.items(): X.iloc[fi_m, X.columns.get_loc(c)] = v
        return pu, len(nmcols)
    def oof_p(mask):
        """XA: classifier probability on the rows in mask, cross-fitted over 4 month groups (no row sees a model trained on its month)."""
        idx = np.where(mask)[0]; mons = sorted(set(m[idx].tolist())); grp = {mm: i % 4 for i, mm in enumerate(mons)}; gi = np.array([grp[x] for x in m[idx]])
        out = np.zeros(len(y))
        for k in range(4):
            tr_k = idx[gi != k]; te_k = idx[gi == k]
            if len(te_k): out[te_k] = fit_cls(tr_k, te_k)
        return np.clip(out, 0.0, 1.0)
    def set_regw(mask):
        ph = np.minimum(oof_p(mask), a.reg_pclip)
        rw = (1.0 - ph) ** (a.reg_pweight if a.reg_pweight > 0 else 2.0)
        ym = y
        if a.reg_mixtarget:
            t_ = (y - ph * ms) / (1.0 - ph); ym = np.where(mask, np.clip(t_, y - a.mixtarget_clip, y + a.mixtarget_clip), y)
        return rw, ym, ph
    if a.rank_p_only:
        _ri = np.where(is_rank)[0]; _pr = fit_cls(np.where(~is_rank)[0], _ri)
        (ROOT / "data/preds").mkdir(exist_ok=True); pl.DataFrame({"MVT_ID_mvt": mvt_id[_ri], "p_raw": _pr}).write_parquet(ROOT / f"data/preds/{a.name}.parquet")
        print(json.dumps(dict(name=a.name, rank_p_only=True, n=int(len(_ri)), p_mean=float(np.mean(_pr)), secs=round(time.time() - t0)))); return
    res_xa = None
    if (a.reg_pweight > 0 or a.reg_mixtarget) and a.strategy == "mixture":
        regw, ymod, _ph = set_regw(trn)
        res_xa = dict(p_train_mean=float(_ph[trn].mean()), w_mean=float(regw[fit_mask].mean()), w_p10=float(np.percentile(regw[fit_mask], 10)), n_fit=int(fit_mask.sum()))
    groups_ = sorted(set(adep)) if a.per_airport else ["ALL"]
    pred = np.full(len(y), np.nan); iters = {}; p_fb = np.full(len(y), np.nan)
    unm = unm_
    unm_ds = (unm_ & cross_) if a.dayshift_midnight else unm_   # rows eligible for the day-shift share
    for g in (groups_ if not a.reuse_taxi else []):
        gm = (adep == g) if g != "ALL" else np.ones(len(y), bool)
        fi = np.where(fit_mask & gm)[0]; vi = np.where(val & gm)[0]
        pred[vi], iters[g] = fit_reg(fi, vi, vi)
    if a.reuse_taxi:
        import glob as _gl
        _tg = a.cv.replace(":", "").replace(",", ""); _fo = {"janjul": "janjul", "months:4,10": "months410", "months:3,9": "months39", "months:2,8": "months28", "months:1,7": "janjul"}.get(a.cv, _tg)
        _c = sorted(_gl.glob(str(ROOT / f"data/oof/{a.reuse_taxi}_{_fo}*.npy"))); assert _c, f"no OOF for {a.reuse_taxi} {_fo}"
        _R = np.load(_c[0]); _o = np.argsort(_R[0]); _vi = np.where(val)[0]; _k = np.searchsorted(_R[0][_o], mvt_id[_vi].astype(float))
        assert np.array_equal(_R[0][_o][_k], mvt_id[_vi].astype(float)), "reuse-taxi: validation IDs do not match"
        pred[_vi] = _R[3][_o][_k]; iters["REUSE"] = _c[0].split("/")[-1]; groups_ = []
    if a.unm_reg:
        fi = np.where(fit_mask & unm)[0]; vi = np.where(val & unm)[0]
        if len(vi) and len(fi) > 500:
            pu, iters["UNM"] = fit_reg(fi, vi, vi); pred[vi] = pu
    if a.unm_masked > 0:
        # unmatched rows: blend in a regressor trained on masked matched rows (same feature pattern, clean labels)
        fi = np.where(fit_mask & (~unm) & (~fb))[0]; vi = np.where(val & unm)[0]
        if len(vi) and len(fi) > 2000:
            pu, _nm = masked_unm_fit(fi, vi, max(200, int(a.fixed_rounds * 0.5) if a.fixed_rounds else 800), trn)
            pred[vi] = (1 - a.unm_masked) * pred[vi] + a.unm_masked * pu
            iters["UNMM"] = int(len(fi)); res_unmm = dict(n_fit=int(len(fi)), n_val=int(len(vi)), n_masked_cols=int(_nm), alpha=a.unm_masked)
    if a.tail_gate > 0 and not a.reuse_taxi:
        # gated tail correction: a second model on the abnormal-NM-offset rows, blended inside the gate only
        _ao = np.nan_to_num(aobt, nan=-1.0); _eo = np.nan_to_num(np.asarray(d_eobt1, dtype=float), nan=0.0)
        gate = (~unm) & (aobt_present) & ((_ao < 300) | (_ao > 2200) | (_eo > 3600))
        if a.tail_gate_lobt:
            _a3l = np.nan_to_num(np.asarray(d_lobt, dtype=float) - _ao, nan=0.0)      # (MVT-LOBT) - (MVT-AOBT_3) = AOBT_3 - LOBT
            gate = gate | ((~unm) & aobt_present & (np.abs(_a3l) > 2400))
            gi = np.where(trn & (~unm) & (~fb) & gate)[0]; gv = np.where(val & gate)[0]
        else:
            gi = np.where(fit_mask & gate)[0]; gv = np.where(val & gate)[0]
        if len(gi) > 2000 and len(gv):
            if a.tail_gate_lobt:
                _off_save = off.copy(); off[:] = np.nan_to_num(np.asarray(d_lobt, dtype=float), nan=0.0)
            _pg, _ = fit_reg(gi, gv, gv, es=False, rounds=max(200, int(a.fixed_rounds * 0.4) if a.fixed_rounds else 800))
            if a.tail_gate_lobt: off[:] = _off_save
            pred[gv] = (1 - a.tail_gate) * pred[gv] + a.tail_gate * _pg
            iters["GATE"] = int(len(gi)); res_gate = dict(n_fit=int(len(gi)), n_val=int(len(gv)), alpha=a.tail_gate)
    if a.strategy == "mixture":
        p_fb[np.where(val)[0]] = fit_cls(np.where(trn)[0], np.where(val)[0])
        pp = np.clip(p_fb[val], 0, 1) ** (1.0 if a.cls_weight in ("mix", "direct", "dmix") else a.p_power)
        if a.cls_tail:
            ov = parse_tail(a.cls_tail); tm = unm[val] & (ms[val] > a.cls_tail_ms); vi_ = np.where(val)[0]
            if tm.sum():
                pt = np.clip(fit_cls(np.where(trn)[0], vi_[tm], ov), 0, 1) ** ov["p_power"]; pp[tm] = pt; p_fb[vi_[tm]] = pt
                res_tail = dict(n=int(tm.sum()), mean_p=float(pt.mean()))
        if a.gate_gamma != 1.0:
            q_ = np.clip(pp, 1e-9, 1 - 1e-9) ** a.gate_gamma; pp = q_ / (q_ + (1 - np.clip(pp, 1e-9, 1 - 1e-9)) ** a.gate_gamma)
        reg_only = pred[val].copy(); art = ms[val].copy()
        if a.dayshift:
            # day-shifted block times: among unmatched LIRF departures with MVT-SCHED > 16 h, a share q of the labels is 86400 + taxi instead of MVT-SCHED
            # share q of day-shifted labels among unmatched departures with MVT-SCHED > 16 h, learned PER AIRPORT on the training rows (>= 5 rows, else 0)
            ds_tr = trn & unm & (y > 80000); t_ds = float(np.median(y[ds_tr] - 86400)) if ds_tr.sum() >= 3 else 941.0   # taxi part of day-shifted labels (regressor output is unreliable on these rows)
            dsm = (dayshift_model_pooled(trn, unm_ds, adep, ms, y, a.dayshift_pooled) if a.dayshift_pooled > 0 else dayshift_model(trn, unm_ds, adep, ms, y))
            _caps = dayshift_caps(trn, unm_ds, adep, ms, y) if a.dayshift_cap < 0 else None
            _cap_v = np.array([_caps[x] for x in adep[val]]) if _caps else a.dayshift_cap
            qv = np.minimum(dayshift_q(dsm, adep[val], ms[val]), _cap_v)
            if a.dayshift_midnight: qv = qv * cross_[val]
            cv_ = unm[val] & (ms[val] > 28800) & (qv > 0.01)
            qq_v = np.zeros(len(art)); qq_v[cv_] = qv[cv_]
            art[cv_] = (1 - qv[cv_]) * ms[val][cv_] + qv[cv_] * (86400 + t_ds); res_ds = dict(caps=(_caps if _caps else a.dayshift_cap), models={k: (round(v[0], 2), round(v[1], 3)) for k, v in dsm.items()}, t_ds=t_ds, n_val=int(cv_.sum()), q_mean=float(qv[cv_].mean()) if cv_.sum() else 0.0)
        if a.mix_additive and a.dayshift:
            if a.mix_additive_mode == "prop":
                _t = np.minimum(pp + qq_v, 1.0) / np.maximum(pp + qq_v, 1e-9); _p, _q = pp * _t, qq_v * _t
            else: _q = qq_v; _p = np.minimum(pp, 1.0 - qq_v)
            pred[val] = _p * ms[val] + _q * (86400 + t_ds) + (1 - _p - _q) * reg_only
        else:
            pred[val] = pp * art + (1 - pp) * reg_only
    e = pred[val] - y[val]; wv = w[val]; av = adep[val]; fbpv = fbp[val]
    def wr(mask): return float(np.sqrt(np.sum((wv * e ** 2)[mask]) / wv[mask].sum()))
    res = dict(name=a.name, args=vars(a), unmm=(res_unmm if a.unm_masked > 0 and "UNMM" in iters else None), n_feats=len(feats), n_val=int(val.sum()), iters=iters, droplift=(res_droplift if a.drop_lift > 0 else None), dayshift=(res_ds if a.dayshift else None), tailcls=(res_tail if a.cls_tail else None),
               xa=res_xa, xlk=(res_xlk if _need_clk else None), rmse_raw=wr(np.ones_like(e, bool)), rmse_cleanpru=wr(~fbpv), rmse_raw_exLIRF=wr(av != "LIRF"), rmse_cleanpru_exLIRF=wr((av != "LIRF") & ~fbpv),
               per_airport_raw={g: float(np.sqrt(np.mean(e[av == g] ** 2))) for g in sorted(set(av))},
               per_airport_cleanpru={g: float(np.sqrt(np.mean(e[(av == g) & ~fbpv] ** 2))) for g in sorted(set(av))}, secs=round(time.time() - t0))
    if a.strategy == "mixture":
        from sklearn.metrics import roc_auc_score
        res["cls_auc"] = float(roc_auc_score(fb[val].astype(int), p_fb[val])); e2 = reg_only - y[val]
        res["rmse_raw_reg_only"] = float(np.sqrt(np.sum(wv * e2 ** 2) / wv.sum()))
    oof = ROOT / "data/oof"; oof.mkdir(exist_ok=True)
    _extra = [reg_only, art] if (a.strategy == "mixture" and "reg_only" in dir() and "art" in dir()) else []   # rows 3-4: taxi branch and art (for Bayes-limit / decomposition analyses); loaders use rows 0-2
    if not _xs.RANK_ONLY: np.save(oof / f"{a.name}_{a.cv.replace(":", "").replace(",", "")}.npy", np.vstack([mvt_id[val], pred[val], p_fb[val]] + _extra))
    _xs.phase("rank")   # SAVE_MODEL
    if a.predict_rank:
        TRN_R = (~is_rank) & ((lk_ & ~unm_) if a.linked_only else ((~lk_ & ~unm_) if a.unlinked_only else True)) & (np.isin(adep, a.only_apt.split(",")) & ~unm_ if a.only_apt else True) & ((hv_ & ~unm_) if a.only_heavy else True)   # lgb_adsb_linked: production training rows
        if (a.reg_pweight > 0 or a.reg_mixtarget) and a.strategy == "mixture": regw, ymod, _ = set_regw(TRN_R)
        ri = np.where(is_rank)[0]; pr = np.full(len(y), np.nan)
        for g in groups_:
            gm = (adep == g) if g != "ALL" else np.ones(len(y), bool)
            fi = np.where(TRN_R & ~drop & gm)[0]; pi = np.where(is_rank & gm)[0]
            rounds = int(iters[g] * 1.1) if iters[g] else 1000
            pr[pi], _ = fit_reg(fi, None, pi, es=False, rounds=rounds)
        if a.unm_reg and "UNM" in iters:
            fi = np.where(TRN_R & ~drop & unm)[0]; pi = np.where(is_rank & unm)[0]
            pr[pi], _ = fit_reg(fi, None, pi, es=False, rounds=int(iters["UNM"] * 1.1) or 300)
        if a.unm_masked > 0 and "UNMM" in iters:
            fi = np.where(TRN_R & ~drop & (~unm) & (~fb))[0]; pi = np.where(is_rank & unm)[0]
            pu, _ = masked_unm_fit(fi, pi, max(200, int(a.fixed_rounds * 0.5) if a.fixed_rounds else 800), ~is_rank)
            pr[pi] = (1 - a.unm_masked) * pr[pi] + a.unm_masked * pu
        if a.tail_gate > 0:
            # ranking-side gated tail correction (mirrors the fold path; was missing before 12 Sep 23:30 -> no submitted file had the gate)
            _ao = np.nan_to_num(aobt, nan=-1.0); _eo = np.nan_to_num(np.asarray(d_eobt1, dtype=float), nan=0.0)
            gate = (~unm) & (aobt_present) & ((_ao < 300) | (_ao > 2200) | (_eo > 3600))
            if a.tail_gate_lobt:
                _a3l = np.nan_to_num(np.asarray(d_lobt, dtype=float) - _ao, nan=0.0)
                gate = gate | ((~unm) & aobt_present & (np.abs(_a3l) > 2400))
                gi = np.where(TRN_R & (~unm) & (~fb) & gate)[0]
            else:
                gi = np.where(TRN_R & ~drop & gate)[0]
            gv = np.where(is_rank & gate)[0]
            if len(gi) > 2000 and len(gv):
                if a.tail_gate_lobt:
                    _off_save = off.copy(); off[:] = np.nan_to_num(np.asarray(d_lobt, dtype=float), nan=0.0)
                _pg, _ = fit_reg(gi, None, gv, es=False, rounds=max(200, int(a.fixed_rounds * 0.4) if a.fixed_rounds else 800))
                if a.tail_gate_lobt: off[:] = _off_save
                pr[gv] = (1 - a.tail_gate) * pr[gv] + a.tail_gate * _pg
                res["rank_gate_n_fit"] = int(len(gi)); res["rank_gate_n_rank"] = int(len(gv))
        if a.strategy == "mixture":
            pr_p = fit_cls(np.where(TRN_R)[0], ri); pp = np.clip(pr_p, 0, 1) ** (1.0 if a.cls_weight in ("mix", "direct", "dmix") else a.p_power)
            if a.cls_tail:
                ov = parse_tail(a.cls_tail); tm = unm[ri] & (ms[ri] > a.cls_tail_ms)
                if tm.sum(): pp[tm] = np.clip(fit_cls(np.where(TRN_R)[0], ri[tm], ov), 0, 1) ** ov["p_power"]; res["rank_tailcls_n"] = int(tm.sum())
            if a.gate_gamma != 1.0:
                q_ = np.clip(pp, 1e-9, 1 - 1e-9) ** a.gate_gamma; pp = q_ / (q_ + (1 - np.clip(pp, 1e-9, 1 - 1e-9)) ** a.gate_gamma)
            art_r = ms[ri].copy()
            if a.dayshift:
                ds_tr = TRN_R & unm & (y > 80000); t_ds = float(np.median(y[ds_tr] - 86400)) if ds_tr.sum() >= 3 else 941.0
                dsm = (dayshift_model_pooled(TRN_R, unm_ds, adep, ms, y, a.dayshift_pooled) if a.dayshift_pooled > 0 else dayshift_model(TRN_R, unm_ds, adep, ms, y))
                _caps = dayshift_caps(TRN_R, unm_ds, adep, ms, y) if a.dayshift_cap < 0 else None
                _cap_r = np.array([_caps[x] for x in adep[ri]]) if _caps else a.dayshift_cap
                qv = np.minimum(dayshift_q(dsm, adep[ri], ms[ri]), _cap_r)
                if a.dayshift_midnight: qv = qv * cross_[ri]
                cr = unm[ri] & (ms[ri] > 28800) & (qv > 0.01)
                qq_r = np.zeros(len(art_r)); qq_r[cr] = qv[cr]
                art_r[cr] = (1 - qv[cr]) * ms[ri][cr] + qv[cr] * (86400 + t_ds); res["rank_dayshift"] = dict(models={k: (round(v[0], 2), round(v[1], 3)) for k, v in dsm.items()}, t_ds=t_ds, n=int(cr.sum()), q_mean=float(qv[cr].mean()) if cr.sum() else 0.0)
            _taxi_r = pr[ri].copy()          # regressor output before mixing -- saved directly, no lossy reconstruction
            if a.mix_additive and a.dayshift:
                if a.mix_additive_mode == "prop":
                    _t = np.minimum(pp + qq_r, 1.0) / np.maximum(pp + qq_r, 1e-9); _p, _q = pp * _t, qq_r * _t
                else: _q = qq_r; _p = np.minimum(pp, 1.0 - qq_r)
                pr[ri] = _p * ms[ri] + _q * (86400 + t_ds) + (1 - _p - _q) * pr[ri]
            else:
                pr[ri] = pp * art_r + (1 - pp) * pr[ri]
            res["rank_p_fb_mean"] = float(pp.mean())
        out = ROOT / "data/preds"; out.mkdir(exist_ok=True)
        _cols = {"MVT_ID_mvt": mvt_id[ri], "TAXITIME_SEC_mvt": pr[ri]}
        if a.strategy == "mixture":
            _cols["p_raw"] = pr_p                    # raw classifier probability, before --p-power
            _cols["art"] = art_r                     # artefact-branch value (schedule-based, incl. day-shift mixing)
            _cols["taxi"] = _taxi_r              # clean-regressor branch, exactly as fitted
            _cols["q_ds"] = (qq_r if a.dayshift else np.zeros(len(ri)))
        pl.DataFrame(_cols).write_parquet(out / f"{a.name}.parquet")
        res["rank_pred_file"] = f"data/preds/{a.name}.parquet"; res["rank_pred_mean"] = float(np.nanmean(pr[ri]))
        _xs.finish(a, res, out / f"{a.name}.parquet")   # SAVE_MODEL
    with open(a.out, "a") as f: f.write(json.dumps(res) + "\n")
    print(json.dumps({k: res[k] for k in res if k in ("name", "rmse_raw", "rmse_cleanpru", "rmse_raw_exLIRF", "cls_auc", "rmse_raw_reg_only", "iters", "n_feats", "secs")}))

if __name__ == "__main__":
    main()
