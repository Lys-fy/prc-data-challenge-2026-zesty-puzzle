# SPDX-License-Identifier: GPL-3.0-only
"""Fixed settings of the baseline. Values marked METHOD come from docs/METHOD.md; the rest are plain defaults, not tuned."""

SEED = 0
# 2025 month-pair folds of docs/METHOD.md §7: a winter and a summer month held out together, like the test periods.
FOLDS: dict[str, tuple[int, int]] = {"1+7": (1, 7), "2+8": (2, 8), "3+9": (3, 9), "4+10": (4, 10), "5+11": (5, 11), "6+12": (6, 12)}
DEFAULT_FOLDS = "1+7,2+8,6+12"

FLOOR_S = 60.0           # METHOD: predictions are floored at 60 s
LOBT_BOUND_S = 3606.0    # METHOD: |y - (MVT - LOBT)| <= 3,606 s on every 2025 row
JOIN_EDGE_S = 3600.0     # METHOD §6: the organisers' flight-plan join edge |BLOCK - LOBT| <= 3,600 s
SCHED_WINDOW_S = 28800.0 # METHOD §6: unmatched rows with MVT - SCHED > 8 h are "exposed" and left to the model
CONGESTION_WINDOW_S = 900.0

NUM_ROUNDS = 600
LGB_PARAMS: dict = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "min_data_per_group": 100, "cat_smooth": 20.0, "seed": SEED, "deterministic": True, "verbose": -1,
}
