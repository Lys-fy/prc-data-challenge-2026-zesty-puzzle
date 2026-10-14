# SPDX-License-Identifier: GPL-3.0-only
"""Regenerate the README figures from the aggregate tables in this folder.

usage: python docs/figures/make_figures.py      (needs matplotlib; reads only the two JSON tables next to this script)

The JSON tables hold aggregates only (shares of rows and squared error, per airport x month RMSE); no competition rows.
"""

import sys

if sys.argv[1:2] in (["-h"], ["--help"]):
    sys.exit(print(__doc__))

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def load(name):
    return json.loads((HERE / name).read_text())


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_row_types():
    rows = load("row_types_2025.json")["rows"]
    names = [f"{r['row_type']}\n(RMSE {r['rmse_s']:,.0f} s)" for r in rows]
    rs = [100 * r["share_rows"] for r in rows]
    es = [100 * r["share_sq_error"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(rows))
    h = 0.38
    ax.barh([i - h / 2 - 0.01 for i in y], rs, height=h, color=BLUE, label="share of departures")
    ax.barh([i + h / 2 + 0.01 for i in y], es, height=h, color=ORANGE, label="share of squared error")
    for i in y:
        ax.text(
            rs[i] + 0.8,
            i - h / 2,
            f"{rs[i]:.2f} %" if rs[i] < 1 else f"{rs[i]:.1f} %",
            va="center",
            fontsize=9,
            color=INK2,
        )
        ax.text(es[i] + 0.8, i + h / 2, f"{es[i]:.1f} %", va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 62)
    ax.set_xlabel("% of 2025 departures / % of squared error (out of fold)")
    clean(ax)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Less than 1 % of the departures carries 44 % of the squared error", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "row_types.png", dpi=110)
    plt.close(fig)


def fig_airport_month():
    d = load("airport_month_rmse_2025.json")
    apts = sorted({c["airport"] for c in d["cells"]})
    months = list(range(1, 13))
    v = {(c["airport"], c["month"]): c["rmse_s"] for c in d["cells"]}
    m = [[v[(a, mo)] for mo in months] for a in apts]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    im = ax.imshow(m, cmap=cmap, vmin=50, vmax=400, aspect="auto")
    for i, a in enumerate(apts):
        for j in range(12):
            x = m[i][j]
            ax.text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if x > 230 else INK)
    ax.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(apts)), apts)
    for j in (0, 1, 5, 6):  # the months of the final ranking set
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, len(apts), fill=False, edgecolor=INK, linewidth=1.2))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("RMSE (s)")
    cb.outline.set_visible(False)
    ax.set_title(
        f"Out-of-fold RMSE on matched 2025 departures, by airport and month (overall {d['overall_rmse_s']} s);"
        "\nframed: the months of the final ranking set",
        loc="left",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(HERE / "airport_month_rmse.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    fig_row_types()
    fig_airport_month()
    for p in sorted(HERE.glob("*.png")):
        print(p.name, p.stat().st_size // 1024, "KB")
