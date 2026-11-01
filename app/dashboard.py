"""streamlit run app/dashboard.py"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from f1strat import charts
from f1strat.model import TyreModel, neutral_pit_losses, pit_loss, pit_losses
from f1strat.openf1 import load_session, season
from f1strat.simulate import simulate
from f1strat.strategy import Plan, actual_stints, pit_window, race_time, search

RESULTS = Path(__file__).resolve().parents[1] / "results"

st.set_page_config(page_title="F1 pit strategy", layout="wide")


@st.cache_data(show_spinner="Loading season")
def races(year: int) -> list[dict]:
    return season(year)


@st.cache_data(show_spinner="Downloading timing data from OpenF1")
def race(session: dict):
    r = load_session(session)
    clean = r.laps[r.laps["clean"]]
    return r, clean, TyreModel.fit(clean)


@st.cache_data(show_spinner="Simulating races")
def monte_carlo(_clean, key, plans, total_laps, green, neutral, draws, driver):
    return simulate(_clean, plans, total_laps, green, neutral, draws=draws, driver=driver)


with st.sidebar:
    st.header("Race")
    year = st.selectbox("Season", [2026, 2025, 2024, 2023], index=2)
    sessions = races(year)
    if not sessions:
        st.stop()
    labels = [f"{s['circuit_short_name']}  ({s['country_name']})" for s in sessions]
    pick = st.selectbox("Grand Prix", range(len(sessions)), format_func=labels.__getitem__,
                        index=min(15, len(sessions) - 1))
    r, clean, model = race(sessions[pick])
    if len(model.compounds) < 2:
        st.error("This race doesn't have two dry compounds with enough clean laps (wet or mixed race).")
        st.stop()

    by_name = {r.drivers.get(d, str(d)): d for d in sorted(model.driver_pace, key=model.driver_pace.get)}
    who = st.selectbox("Plan on the pace of", ["Field median", *by_name])
    driver = by_name.get(who)
    pace = model.reference_pace(driver)

    measured = pit_loss(r.laps, model)
    loss = st.slider("Pit loss (s)", 12.0, 35.0, round(measured or 22.0, 1), 0.5,
                     help="Starts at the 40th percentile of this race's green-flag stops.")
    draws = st.select_slider("Simulated races", [200, 500, 1000, 2000], value=500)

st.title(r.name)
plans = search(model, r.total_laps, loss, pace)
best = plans[0]

rows: list[tuple[str, Plan, float]] = [(f"{p.label}", p, p.time - best.time) for p in plans[:5]]
win_gap = None
if r.finish_order:
    w = r.finish_order[0]
    st_w = actual_stints(r.laps, w, r.total_laps)
    if st_w and all(s.compound in model.compounds for s in st_w):
        t = race_time(model, pace, st_w, loss)
        win_gap = t - best.time
        rows.append((f"{r.drivers.get(w, w)} (winner, actual)", Plan(tuple(st_w), t), win_gap))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Laps", r.total_laps)
c2.metric("Pit loss", f"{loss:.1f} s")
c3.metric("Fastest plan", f"{best.label}, lap {', '.join(map(str, best.pit_laps))}")
c4.metric("Winner vs fastest plan", "n/a" if win_gap is None else f"{win_gap:+.1f} s")
source = "measured from this race" if measured and abs(loss - measured) < 0.05 else "set by hand"
st.caption(f"{len(clean)} clean laps · pit loss {source}"
           f" · {len(r.neutralised)} laps under safety car or VSC")

tab_plan, tab_tyre, tab_mc, tab_bt = st.tabs(["Strategy", "Tyres", "Monte Carlo", "Backtest"])

with tab_plan:
    windows = {p.label: pit_window(model, p, r.total_laps, loss, pace) for p in plans[:5]}
    st.plotly_chart(charts.timelines(rows, r.total_laps, windows), use_container_width=True)
    st.caption("Black bands: laps where each stop can move and keep the plan within 1 s of its best. "
               "Every plan is the best one for its set of tyres; order of compounds doesn't change the time.")
    table = pd.DataFrame([{
        "plan": p.label, "stops": p.stops, "pit laps": ", ".join(map(str, p.pit_laps)),
        "window": ", ".join(f"{a}-{b}" for a, b in windows.get(p.label, [])),
        "gap (s)": round(p.time - best.time, 1)} for p in plans[:5]])
    st.dataframe(table, hide_index=True, use_container_width=True)

with tab_tyre:
    left, right = st.columns([2, 1])
    left.plotly_chart(charts.degradation(clean, model), use_container_width=True)
    right.dataframe(pd.DataFrame([{
        "compound": c.title(), "vs " + model.compounds[0].lower(): f"{model.offset[c]:+.2f} s",
        "wear": f"{model.deg[c]:+.3f} s/lap", "longest stint": model.max_age[c]} for c in model.compounds]),
        hide_index=True, use_container_width=True)
    right.markdown(f"Fuel: **{model.fuel:+.3f} s/lap** as the car gets lighter.\n\n"
                   "Wear is the slope of the line. A straight line can't see a cliff, so plans may run "
                   "at most 5 laps past the longest stint the data shows.")

with tab_mc:
    green = pit_losses(r.laps, model) or [loss]
    if measured and abs(loss - measured) > 0.05:
        green = [g + loss - measured for g in green]  # shift the measured spread to the chosen loss
    under = neutral_pit_losses(r.laps, r.neutralised)
    neutral = float(np.median(under)) if len(under) >= 3 else None
    out = monte_carlo(clean, (r.name, loss, who), plans[:6], r.total_laps, green, neutral, draws, driver)
    left, right = st.columns(2)
    left.subheader("How often each plan is fastest")
    left.plotly_chart(charts.win_share(out), use_container_width=True)
    right.subheader("Time lost to the best plan of each race")
    right.plotly_chart(charts.gap_spread(out), use_container_width=True)
    st.caption("Each simulated race refits the tyre model on a bootstrap of stints, draws a pit loss "
               "from this race's stops and adds safety cars at the 2023-2026 rate. A plan pits early "
               "under a safety car when that saves time.")

with tab_bt:
    path = RESULTS / "backtest.csv"
    if not path.exists():
        st.info("Run `f1strat backtest 2023 2024 2025 2026` to fill this tab.")
    else:
        df = pd.read_csv(path, keep_default_na=False, na_values=[""])
        df["skip"] = df["skip"].fillna("")
        summary = pd.read_csv(RESULTS / "summary.csv")
        st.dataframe(summary, hide_index=True, use_container_width=True)
        pred = st.radio("Predictor", ["forecast", "model", "monte", "lastyear"], horizontal=True)
        st.plotly_chart(charts.backtest_scatter(df, pred), use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
