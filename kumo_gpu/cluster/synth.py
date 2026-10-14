# SPDX-License-Identifier: GPL-3.0-only
"""KUMO_CLUSTER_SETUP optional synthetic query rows for distillation (Nums AI 2610.01435 / GEAR style: perturbed real rows).
make(X, qi, frac, seed, key) -> (Xs, base_idx, None): n = round(frac * len(qi)) synthetic rows; each starts from a random real query row of the
unit and replaces a random 30 % of its 412 features with the values of other random query rows of the same unit (column-wise marginal swap;
features only, no labels are used). Deterministic in (seed, key), so a CPU student script can regenerate the exact rows from the staged frame
with the same call (positions are relative to the per-airport frame km_<APT>.parquet)."""

import zlib, numpy as np


def make(X, qi, frac, seed=0, key="", pswap=0.30):
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    n = int(round(frac * len(qi)))
    base = rng.choice(qi, n, replace=True)
    Xs = X[base].copy()
    mask = rng.random((n, X.shape[1])) < pswap
    donors = rng.choice(qi, (n, X.shape[1]), replace=True)
    r, c = np.nonzero(mask)
    Xs[r, c] = X[donors[r, c], c]
    return Xs, base, None
