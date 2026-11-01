from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import TyreModel
from .strategy import Plan

COLOR = {"SOFT": "#e34948", "MEDIUM": "#eda100", "HARD": "#2a78d6"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def figure(race_name: str, laps, model: TyreModel, rows: list[tuple[str, Plan, float]],
           total_laps: int, path: str) -> None:
    """Left: tyre degradation with fuel and car pace removed. Right: plan timelines."""
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # Degradation: what's left of each lap once car pace and fuel are taken out.
    d = laps[laps["compound"].isin(model.compounds)]
    pace = d["driver_number"].map(model.driver_pace)
    resid = d["lap_duration"] - pace - model.fuel * d["lap_number"]
    for c in model.compounds:
        m = d["compound"] == c
        ax1.scatter(d.loc[m, "tyre_age"], resid[m], s=9, color=COLOR[c], alpha=0.25, linewidths=0)
        age = np.arange(0, model.max_age[c] + 1)
        ax1.plot(age, model.offset[c] + model.deg[c] * age, color=COLOR[c], lw=2,
                 label=f"{c.title()}  {model.deg[c]:+.3f} s/lap")
    lo, hi = np.percentile(resid, [1, 99])
    ax1.set_ylim(lo - 0.3, hi + 0.3)
    ax1.set_xlabel("Tyre age (laps)")
    ax1.set_ylabel("Lap time, car pace and fuel removed (s)")
    ax1.set_title("Tyre degradation", loc="left", fontweight="bold")
    ax1.grid(axis="y", color=GRID, lw=0.8)
    ax1.legend(frameon=False, loc="upper left")

    # Strategy timelines, fastest at the top.
    for i, (name, plan, gap) in enumerate(rows):
        y = len(rows) - 1 - i
        for s in plan.stints:
            ax2.barh(y, s.laps - 0.3, left=s.first_lap - 1 + 0.15, height=0.62,
                     color=COLOR[s.compound], edgecolor=SURFACE, linewidth=2)
            if s.laps >= 4:
                ax2.text(s.first_lap - 1 + s.laps / 2, y, s.compound[0], ha="center",
                         va="center", color=INK, fontsize=9, fontweight="bold")
        ax2.text(total_laps + 1, y, "fastest" if gap == 0 else f"+{gap:.1f} s",
                 va="center", color=MUTED)
    ax2.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax2.set_xlim(0, total_laps + 8)
    ax2.set_xlabel("Lap")
    ax2.set_title("Best plan per tyre set vs. what the winner did", loc="left", fontweight="bold")
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)

    fig.suptitle(race_name, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
