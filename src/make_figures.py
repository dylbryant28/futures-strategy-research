"""README figures.
  figures/core_equity.png    : cumulative net return of the core strategy, 2010-2026, holdout shaded
  figures/what_survived.png  : profit factor 2015-22 vs 2023-26 for every variant in research rounds 4-9"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from holdout_eval import trades  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG, RES = ROOT / "figures", ROOT / "results"
BLUE, ORANGE, GRAY, INK, INK2, SURF = "#2a78d6", "#eb6834", "#b4b2ab", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRAY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def core_equity():
    t, _ = trades(["NQ2010", "NQ2015", "NQ"])
    nz = t[t.strat == "noise"].sort_values("ts")
    bps = (nz.gross - 1.0) / nz.price * 1e4  # net of a 4-tick round trip, basis points of notional
    cum = bps.cumsum() / 100  # percent, 1 contract-equivalent, no compounding
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    ax.axvspan(pd.Timestamp("2010-06-01"), pd.Timestamp("2015-01-01"), color="#f0efec", zorder=0)
    ax.text(pd.Timestamp("2012-09-01"), cum.max() * 0.92, "holdout\n(untouched until final config)", ha="center",
            va="top", color=INK2, fontsize=9)
    ax.plot(nz.ts, cum, color=BLUE, lw=2)
    ax.axhline(0, color=GRAY, lw=1)
    ax.set_ylabel("cumulative return, % (simple sum)")
    ax.set_title("Core strategy: Nasdaq noise-area momentum, net of costs", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", color="#e5e4df", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "core_equity.png")


def what_survived():
    frames = []
    for n in range(4, 10):
        x = pd.read_csv(RES / f"research{n}.csv")
        frames.append(x[["pf_2015-22", "pf_2023-26"]].assign(round=n))
    x = pd.concat(frames).dropna()
    x = x.clip(upper=2.0)
    ok = (x["pf_2015-22"] > 1) & (x["pf_2023-26"] > 1)
    fig, ax = plt.subplots(figsize=(6, 5.2), dpi=150)
    ax.axvline(1, color=GRAY, lw=1)
    ax.axhline(1, color=GRAY, lw=1)
    ax.scatter(x.loc[~ok, "pf_2015-22"], x.loc[~ok, "pf_2023-26"], s=10, color=GRAY, alpha=0.6, lw=0,
               label=f"fails a period ({(~ok).sum()})")
    ax.scatter(x.loc[ok, "pf_2015-22"], x.loc[ok, "pf_2023-26"], s=12, color=BLUE, alpha=0.8, lw=0,
               label=f"profitable in both ({ok.sum()})")
    ax.scatter([1.34], [1.30], s=90, color=ORANGE, edgecolor=SURF, lw=2, zorder=5, label="core strategy")
    ax.annotate("core strategy", (1.34, 1.30), xytext=(1.48, 1.62), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("profit factor 2015-22")
    ax.set_ylabel("profit factor 2023-26")
    ax.set_title(f"{len(x)} variants from research rounds 4-9", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0, 2.05)
    fig.tight_layout()
    fig.savefig(FIG / "what_survived.png")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    what_survived()
    core_equity()
