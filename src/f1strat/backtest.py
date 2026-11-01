"""Check the planner against what teams actually did, race by race.

For every dry race the field's real strategy is reduced to two numbers: the
most common stop count among classified finishers and the median lap of their
first stop. Four predictors try to match them:

- model:     the deterministic search, fitted on the same race (in-sample)
- monte:     the plan that won most Monte Carlo draws, same race
- forecast:  the search fitted on last year's race at the same circuit, which is
             the only honest pre-race test since it never sees this race's laps
- last year: copy what the field did at this circuit a year earlier
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .model import TyreModel, evaluate, neutral_pit_losses, pit_loss, pit_losses
from .openf1 import DRY, Race, load_session, season
from .simulate import simulate
from .strategy import actual_stints, race_time, search

DEFAULT_PIT_LOSS = 22.0


@dataclass
class Row:
    year: int
    circuit: str
    laps: int
    skip: str = ""
    neutral_periods: int = 0
    neutral_laps: int = 0
    green_loss: float = np.nan
    neutral_loss: float = np.nan
    field_stops: float = np.nan
    field_share: float = np.nan     # share of finishers on the most common stop count
    field_first: float = np.nan     # median first-stop lap among them
    model_stops: float = np.nan
    model_first: float = np.nan
    monte_stops: float = np.nan
    monte_first: float = np.nan
    monte_wins: float = np.nan
    forecast_stops: float = np.nan
    forecast_first: float = np.nan
    lastyear_stops: float = np.nan
    lastyear_first: float = np.nan
    winner: str = ""
    winner_gap: float = np.nan      # winner's real strategy vs the best plan, same model
    mae_stints: float = np.nan      # lap-time error on held-out stints, this model
    mae_stints_gbm: float = np.nan  # same split, gradient boosting
    mae_old: float = np.nan         # trained on young tyres, tested on old ones
    mae_old_gbm: float = np.nan


def field_strategy(race: Race) -> tuple[int, float, float] | None:
    """(most common stop count, its share, median first-stop lap) of dry-tyre finishers.

    A tyre change in the first or last three laps doesn't count: that is a
    lap-1 red flag, a puncture or a fastest-lap stop, not a strategy. It also
    drops the one-lap stints the feed sometimes invents behind a final safety car.
    """
    stops, firsts = [], {}
    for drv in race.finish_order:
        st = actual_stints(race.laps, drv, race.total_laps)
        if not st or any(s.compound not in DRY for s in st):
            continue
        pits = [s.last_lap for s in st[:-1] if 3 < s.last_lap < race.total_laps - 3]
        stops.append(len(pits))
        if pits:
            firsts.setdefault(len(pits), []).append(pits[0])
    if len(stops) < 5:
        return None
    mode, count = Counter(stops).most_common(1)[0]
    first = float(np.median(firsts[mode])) if mode > 0 else np.nan
    return mode, count / len(stops), first


def periods(laps: list[int]) -> list[int]:
    """Lengths of runs of consecutive laps, e.g. [10, 11, 12, 30] -> [3, 1]."""
    runs: list[int] = []
    for i, n in enumerate(laps):
        if i and n == laps[i - 1] + 1:
            runs[-1] += 1
        else:
            runs.append(1)
    return runs


def check_race(race: Race, prior: dict | None, draws: int = 300,
               lap_model: bool = True) -> tuple[Row, dict | None]:
    row = Row(race.year, race.circuit, race.total_laps)
    neutral = race.neutralised
    row.neutral_periods, row.neutral_laps = len(periods(neutral)), len(neutral)
    clean = race.laps[race.laps["clean"]]
    actual = field_strategy(race)
    if actual is None:
        row.skip = "fewer than 5 finishers on dry tyres only"
        return row, None
    row.field_stops, row.field_share, row.field_first = actual
    if prior is not None:
        row.lastyear_stops, row.lastyear_first = prior["field_stops"], prior["field_first"]
        p = search(prior["model"], race.total_laps, prior["loss"])
        if p:
            row.forecast_stops, row.forecast_first = p[0].stops, p[0].pit_laps[0]

    if len(clean) < 300:
        row.skip = f"only {len(clean)} clean laps"
        return row, None
    model = TyreModel.fit(clean)
    if len(model.compounds) < 2:
        row.skip = "fewer than two dry compounds with enough laps"
        return row, None
    measured = pit_losses(race.laps, model)
    loss = pit_loss(race.laps, model) or DEFAULT_PIT_LOSS
    row.green_loss = loss
    under = neutral_pit_losses(race.laps, neutral)
    row.neutral_loss = float(np.median(under)) if under else np.nan
    plans = search(model, race.total_laps, loss)
    if not plans:
        row.skip = "no plan fits the stint limits"
        return row, None
    row.model_stops, row.model_first = plans[0].stops, plans[0].pit_laps[0]

    mc = simulate(clean, plans, race.total_laps, measured if len(measured) >= 3 else [loss],
                  draws=draws, models=60)
    row.monte_stops, row.monte_first, row.monte_wins = mc[0].plan.stops, mc[0].plan.pit_laps[0], mc[0].wins

    if race.finish_order:
        drv = race.finish_order[0]
        st = actual_stints(race.laps, drv, race.total_laps)
        row.winner = race.drivers.get(drv, str(drv))
        if st and all(s.compound in model.compounds for s in st):
            row.winner_gap = race_time(model, model.reference_pace(), st, loss) - plans[0].time

    if lap_model:
        e = evaluate(clean)
        row.mae_stints, row.mae_stints_gbm = e["interp_phys"], e["interp_gbm"]
        row.mae_old, row.mae_old_gbm = e["extrap_phys"], e["extrap_gbm"]

    state = {"year": race.year, "model": model, "loss": loss,
             "field_stops": row.field_stops, "field_first": row.field_first}
    return row, state


def run(years: list[int], draws: int = 300, lap_model: bool = True, log=print) -> pd.DataFrame:
    rows, prior = [], {}
    for year in sorted(years):
        for s in season(year):
            try:
                race = load_session(s)
                last = prior.get(race.circuit)
                last = last if last and last["year"] == year - 1 else None
                row, state = check_race(race, last, draws, lap_model)
            except ValueError as e:  # a race that wasn't run, or a feed with nothing usable
                row, state = Row(year, s["circuit_short_name"], 0, skip=str(e)), None
            if state is not None:
                prior[row.circuit] = state
            rows.append(asdict(row))
            log(f"{year} {row.circuit:<16} {row.skip or 'ok'}")
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Stop-count hit rate and first-stop error for each predictor."""
    ok = df[df["skip"] == ""]
    out = []
    for name in ("model", "monte", "forecast", "lastyear", "one_stop"):
        if name == "one_stop":
            stops = pd.Series(1, index=ok.index)
            first = (ok["laps"] / 2).round()
        else:
            stops, first = ok[f"{name}_stops"], ok[f"{name}_first"]
        has = stops.notna()
        hit = stops[has] == ok.loc[has, "field_stops"]
        both = has & first.notna() & ok["field_first"].notna()
        err = (first[both] - ok.loc[both, "field_first"]).abs()
        out.append({"predictor": name, "races": int(has.sum()), "stops_right": float(hit.mean()),
                    "first_stop_mae": float(err.mean()), "first_stop_within_3": float((err <= 3).mean())})
    return pd.DataFrame(out)
