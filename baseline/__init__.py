# SPDX-License-Identifier: GPL-3.0-only
"""Laptop-scale baseline for the PRC Data Challenge 2026: one LightGBM model on the residual to MVT - AOBT_3."""
import os

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")  # the files carry an R vctrs tag
