"""Brute-force search over one- and two-stop strategies.

A stint's time has a closed form under the linear model, so every plan for a
70-lap race (about 40,000 of them) is scored in under 0.3 s.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .model import TyreModel

HARDNESS = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class Stint:
    compound: str
    first_lap: int
    last_lap: int
    start_age: int = 0

    @property
    def laps(self) -> int:
        return self.last_lap - self.first_lap + 1


@dataclass(frozen=True)
class Plan:
    stints: tuple[Stint, ...]
    time: float  # race time in seconds, pit losses included

    @property
    def stops(self) -> int:
        return len(self.stints) - 1

    @property
    def label(self) -> str:
        return "-".join(s.compound[0] for s in self.stints)

    @property
    def pit_laps(self) -> tuple[int, ...]:
        return tuple(s.last_lap for s in self.stints[:-1])


def stint_time(m: TyreModel, pace: float, s: Stint) -> float:
    n, a0, l0 = s.laps, s.start_age, s.first_lap
    tri = n * (n - 1) / 2  # sum of 0..n-1
    return (n * (pace + m.offset[s.compound])
            + m.fuel * (n * l0 + tri)
            + m.deg[s.compound] * (n * a0 + tri))


def race_time(m: TyreModel, pace: float, stints, pit_loss: float) -> float:
    return sum(stint_time(m, pace, s) for s in stints) + pit_loss * (len(stints) - 1)


def search(m: TyreModel, total_laps: int, pit_loss: float, pace: float | None = None,
           max_stops: int = 2, min_stint: int = 5, stretch: int = 5) -> list[Plan]:
    """Best plan for every compound sequence, fastest first.

    A stint may run at most `stretch` laps past the longest stint the model saw
    on that compound; beyond that a linear fit has no evidence of a cliff.
    """
    pace = m.reference_pace() if pace is None else pace
    cap = {c: m.max_age[c] + stretch for c in m.compounds}
    best: dict[tuple[str, ...], tuple] = {}
    for stops in range(1, max_stops + 1):
        for seq in product(m.compounds, repeat=stops + 1):
            if len(set(seq)) < 2:  # dry-race rule: two different compounds
                continue
            for pits in _pit_laps(total_laps, stops, min_stint):
                bounds = (0, *pits, total_laps)
                stints = tuple(Stint(c, bounds[i] + 1, bounds[i + 1]) for i, c in enumerate(seq))
                if any(s.laps > cap[s.compound] for s in stints):
                    continue
                t = race_time(m, pace, stints, pit_loss)
                # The model has no fuel-by-compound term, so M-H and H-M cost the
                # same. Keep one per tyre set, softest first (the usual call,
                # since it buys track position at the start).
                key = tuple(sorted(seq, key=HARDNESS.index))
                rank = (round(t, 6), [HARDNESS.index(c) for c in seq])
                if key not in best or rank < best[key][0]:
                    best[key] = (rank, Plan(stints, t))
    return sorted((p for _, p in best.values()), key=lambda p: p.time)


def pit_window(m: TyreModel, plan: Plan, total_laps: int, pit_loss: float,
               pace: float | None = None, tolerance: float = 1.0,
               min_stint: int = 5, stretch: int = 5) -> list[tuple[int, int]]:
    """Range of laps for each stop that keeps the plan within `tolerance` s of its best.

    Moves one stop at a time and holds the others, under the same stint limits
    as `search`.
    """
    pace = m.reference_pace() if pace is None else pace
    cap = {c: m.max_age[c] + stretch for c in m.compounds}
    seq = [s.compound for s in plan.stints]
    windows = []
    for k in range(plan.stops):
        ok = []
        for lap in range(1, total_laps):
            pits = list(plan.pit_laps)
            pits[k] = lap
            bounds = (0, *pits, total_laps)
            stints = [Stint(c, bounds[i] + 1, bounds[i + 1]) for i, c in enumerate(seq)]
            if any(s.laps < min_stint or s.laps > cap[s.compound] for s in stints):
                continue
            if race_time(m, pace, stints, pit_loss) <= plan.time + tolerance:
                ok.append(lap)
        windows.append((min(ok), max(ok)))
    return windows


def actual_stints(laps: pd.DataFrame, driver: int, total_laps: int | None = None) -> list[Stint]:
    d = laps[laps["driver_number"] == driver].dropna(subset=["stint"])
    out = []
    for _, g in d.groupby("stint"):
        g = g.sort_values("lap_number")
        out.append(Stint(g["compound"].iloc[0], int(g["lap_number"].iloc[0]),
                         int(g["lap_number"].iloc[-1]), int(g["tyre_age"].iloc[0])))
    if out and total_laps:  # the stints feed can stop a lap short of the flag
        last = out[-1]
        out[-1] = Stint(last.compound, last.first_lap, total_laps, last.start_age)
    return out


def _pit_laps(total: int, stops: int, min_stint: int):
    if stops == 1:
        for a in range(min_stint, total - min_stint + 1):
            yield (a,)
    else:
        for a in range(min_stint, total - 2 * min_stint + 1):
            for b in range(a + min_stint, total - min_stint + 1):
                yield (a, b)
