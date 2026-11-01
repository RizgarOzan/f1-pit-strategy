"""Plotly figures for the dashboard. Same colours as the static PNG charts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .model import TyreModel
from .plot import COLOR, GRID, INK, MUTED
from .simulate import Outcome
from .strategy import Plan

FONT = dict(family="Inter, Segoe UI, sans-serif", size=13, color=INK)


def _style(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(height=height, font=FONT, plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=30, b=10), **layout)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def degradation(laps: pd.DataFrame, model: TyreModel) -> go.Figure:
    """Each clean lap with car pace and fuel taken out, plus the fitted wear line."""
    d = laps[laps["compound"].isin(model.compounds)]
    resid = d["lap_duration"] - d["driver_number"].map(model.driver_pace) - model.fuel * d["lap_number"]
    fig = go.Figure()
    for c in model.compounds:
        m = d["compound"] == c
        fig.add_scatter(x=d.loc[m, "tyre_age"], y=resid[m], mode="markers", name=c.title(),
                        marker=dict(color=COLOR[c], size=5, opacity=0.3), legendgroup=c, showlegend=False,
                        hovertemplate="age %{x}<br>%{y:.2f} s<extra></extra>")
        age = np.arange(0, model.max_age[c] + 1)
        fig.add_scatter(x=age, y=model.offset[c] + model.deg[c] * age, mode="lines", legendgroup=c,
                        name=f"{c.title()}  {model.deg[c]:+.3f} s/lap", line=dict(color=COLOR[c], width=3))
    lo, hi = np.percentile(resid, [1, 99])
    fig.update_yaxes(range=[lo - 0.3, hi + 0.3], title="Lap time, car pace and fuel removed (s)")
    fig.update_xaxes(title="Tyre age (laps)")
    return _style(fig, 430, legend=dict(orientation="h", y=1.08, x=0))


def timelines(rows: list[tuple[str, Plan, float]], total_laps: int,
              windows: dict[str, list[tuple[int, int]]] | None = None) -> go.Figure:
    """One bar per plan, a block per stint, fastest at the top. Pit windows as thin bands."""
    fig = go.Figure()
    names = [r[0] for r in rows]
    for name, plan, gap in rows:
        for s in plan.stints:
            fig.add_bar(y=[name], x=[s.laps], base=[s.first_lap - 1], orientation="h",
                        marker=dict(color=COLOR[s.compound], line=dict(color="white", width=2)),
                        text=s.compound[0] if s.laps >= 4 else "", textposition="inside",
                        insidetextanchor="middle", showlegend=False,
                        hovertemplate=f"{s.compound.title()}: laps {s.first_lap}-{s.last_lap}<extra></extra>")
        fig.add_annotation(y=name, x=total_laps + 1, xanchor="left", showarrow=False,
                           text="fastest" if gap == 0 else f"+{gap:.1f} s", font=dict(color=MUTED))
        for lo, hi in (windows or {}).get(name, []):
            fig.add_shape(type="rect", x0=lo - 0.5, x1=hi + 0.5, y0=names.index(name) + 0.42,
                          y1=names.index(name) + 0.5, fillcolor=INK, opacity=0.6, line_width=0)
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=names, gridcolor="white")
    fig.update_xaxes(range=[0, total_laps + 8], title="Lap")
    return _style(fig, 90 + 52 * len(rows), barmode="overlay", bargap=0.35)


def win_share(outcomes: list[Outcome]) -> go.Figure:
    labels = [f"{o.plan.label}  ({', '.join(map(str, o.plan.pit_laps))})" for o in outcomes]
    fig = go.Figure(go.Bar(x=[o.wins for o in outcomes], y=labels, orientation="h",
                           marker_color=INK, text=[f"{o.wins:.0%}" for o in outcomes],
                           textposition="outside", hovertemplate="%{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.1], title="Share of simulated races won")
    return _style(fig, 80 + 44 * len(outcomes))


def gap_spread(outcomes: list[Outcome]) -> go.Figure:
    """How far behind the best plan of each draw every plan finished."""
    fig = go.Figure()
    for o in outcomes:
        fig.add_box(x=o.gaps, name=f"{o.plan.label} ({', '.join(map(str, o.plan.pit_laps))})",
                    marker_color=INK, boxpoints=False, line_width=1.5)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Seconds behind the fastest plan of that race", rangemode="tozero")
    return _style(fig, 80 + 44 * len(outcomes), showlegend=False)


def backtest_scatter(df: pd.DataFrame, predictor: str = "forecast") -> go.Figure:
    """Predicted vs real first-stop lap, one dot per race."""
    d = df[(df["skip"] == "") & df[f"{predictor}_first"].notna() & df["field_first"].notna()]
    hit = d[f"{predictor}_stops"] == d["field_stops"]
    fig = go.Figure()
    for ok, name, color in ((True, "stop count right", INK), (False, "stop count wrong", COLOR["SOFT"])):
        g = d[hit == ok]
        fig.add_scatter(x=g["field_first"], y=g[f"{predictor}_first"], mode="markers", name=name,
                        marker=dict(color=color, size=9, opacity=0.8),
                        text=g["year"].astype(str) + " " + g["circuit"],
                        hovertemplate="%{text}<br>real %{x}, predicted %{y}<extra></extra>")
    top = float(max(d["field_first"].max(), d[f"{predictor}_first"].max())) + 3 if len(d) else 40
    fig.add_scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=MUTED, dash="dot"), showlegend=False)
    fig.update_xaxes(title="Real first stop (median lap of the field)", range=[0, top])
    fig.update_yaxes(title="Predicted first stop", range=[0, top])
    return _style(fig, 460, legend=dict(orientation="h", y=1.08, x=0))
