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


PREDICTORS = {"model": "Search, same race", "monte": "Monte Carlo, same race",
              "forecast": "Search, last year's race", "lastyear": "Copy last year's field",
              "one_stop": "Always one stop at half distance"}


def _axes(ncols: int, size, ratios=None):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, ncols, figsize=size, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": ratios} if ratios else None)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return fig, axes


def backtest_figure(df, summary, path: str) -> None:
    """Left: how often each predictor called the field's stop count. Right: first-stop lap, model vs reality."""
    fig, (ax1, ax2) = _axes(2, (13, 5.2), [1.15, 1])
    s = summary.set_index("predictor").loc[list(PREDICTORS)]
    y = np.arange(len(s))[::-1]
    ax1.barh(y, s["stops_right"], height=0.55, color=[INK, INK, INK, MUTED, GRID])
    for yi, (name, r) in zip(y, s.iterrows()):
        ax1.text(r["stops_right"] + 0.01, yi, f"{r['stops_right']:.0%}  ({int(r['races'])} races)",
                 va="center", color=MUTED)
    ax1.set_yticks(y, [PREDICTORS[n] for n in s.index])
    ax1.set_xlim(0, 1.15)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xlabel("Races where the predicted stop count matched the field's most common one")
    ax1.set_title("Stop count", loc="left", fontweight="bold")
    ax1.tick_params(axis="y", length=0)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)

    d = df[(df["skip"] == "") & df["model_first"].notna() & df["field_first"].notna()]
    hit = d["model_stops"] == d["field_stops"]
    ax2.scatter(d.loc[hit, "field_first"], d.loc[hit, "model_first"], s=30, color=INK,
                label="stop count right")
    ax2.scatter(d.loc[~hit, "field_first"], d.loc[~hit, "model_first"], s=30, facecolor="none",
                edgecolor=COLOR["SOFT"], lw=1.5, label="stop count wrong")
    top = max(d["field_first"].max(), d["model_first"].max()) + 3
    ax2.plot([0, top], [0, top], color=MUTED, lw=1, ls=":")
    ax2.set_xlim(0, top)
    ax2.set_ylim(0, top)
    ax2.set_xlabel("Real first stop (median lap of the field)")
    ax2.set_ylabel("Predicted first stop (search, same race)")
    ax2.set_title("First stop lap", loc="left", fontweight="bold")
    ax2.grid(color=GRID, lw=0.8)
    ax2.legend(frameon=False, loc="upper left")
    fig.suptitle(f"Backtest: {len(df[df['skip'] == ''])} dry races, {int(df['year'].min())}-{int(df['year'].max())}",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def monte_carlo_figure(race_name: str, outcomes, path: str) -> None:
    """Left: share of simulated races each plan won. Right: spread of time lost to the best plan."""
    fig, (ax1, ax2) = _axes(2, (13, 0.9 + 0.55 * len(outcomes) + 1.2))
    labels = [f"{o.plan.label}  ({', '.join(map(str, o.plan.pit_laps))})" for o in outcomes]
    y = np.arange(len(outcomes))[::-1]
    ax1.barh(y, [o.wins for o in outcomes], height=0.55, color=INK)
    for yi, o in zip(y, outcomes):
        ax1.text(o.wins + 0.01, yi, f"{o.wins:.0%}", va="center", color=MUTED)
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, 1.1)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xlabel("Share of simulated races won")
    ax1.set_title("How often each plan is fastest", loc="left", fontweight="bold")
    ax1.tick_params(axis="y", length=0)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)

    ax2.boxplot([o.gaps for o in outcomes], positions=y, vert=False, widths=0.5, showfliers=False,
                medianprops=dict(color=COLOR["SOFT"], lw=2), boxprops=dict(color=INK),
                whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax2.set_yticks(y, [""] * len(y))
    ax2.set_xlim(left=0)
    ax2.set_xlabel("Seconds behind the fastest plan of that race (box: middle half, whiskers: 1.5 IQR)")
    ax2.set_title("Time lost when it isn't", loc="left", fontweight="bold")
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    fig.suptitle(f"{race_name}: Monte Carlo", x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
