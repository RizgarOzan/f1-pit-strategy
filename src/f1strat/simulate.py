"""Monte Carlo: how often does each plan come out fastest once the race gets messy?

The search in `strategy.py` treats the fitted numbers as exact and the race as
green from lights to flag. Each draw here changes three things:

- the tyre model, refit on a bootstrap resample of whole stints, so a compound
  with few laps gets a wide spread of wear rates;
- the green-flag pit loss, drawn from the stops measured in the race;
- safety cars and VSCs, drawn at the rate seen across 2023-2026.

A plan reacts to a neutralisation the way a pit wall does: if moving the next
stop onto that lap costs less in tyre time than the cheaper stop saves, the
car pits then. Lap-to-lap noise is left out on purpose: it hits every plan the
same way and would only blur the comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import TyreModel
from .strategy import Plan, Stint, race_time

# Measured over 79 dry races, 2023-2026, by `f1strat backtest` (see docs/validation.md):
# 1.19 neutralised periods per race, 2.8 laps long on average, and a stop under
# one cost a median 66% of a green-flag stop.
NEUTRAL_PER_RACE = 1.2
NEUTRAL_LENGTHS = (1, 2, 3, 4, 5)  # laps, drawn uniformly (mean 3)
NEUTRAL_PIT_SHARE = 0.66


@dataclass
class Outcome:
    plan: Plan
    gaps: np.ndarray  # race time minus the fastest candidate, per draw
    wins: float       # share of draws where this plan was fastest
    reacted: float    # share of draws where a safety car moved a stop

    @property
    def mean_gap(self) -> float:
        return float(self.gaps.mean())


def bootstrap_models(laps: pd.DataFrame, n: int, rng: np.random.Generator) -> list[TyreModel]:
    """Refit the model on `n` resamples of whole stints (laps of a stint stay together)."""
    keys = laps["driver_number"].astype(str) + "-" + laps["stint"].astype(str)
    groups = [g for _, g in laps.groupby(keys)]
    wanted = set(TyreModel.fit(laps).compounds)
    models = []
    while len(models) < n:
        pick = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in pick], ignore_index=True)
        # Resampling can drop a whole compound; keep only draws that model all of them.
        m = TyreModel.fit(sample)
        if set(m.compounds) == wanted:
            models.append(m)
    return models


def neutral_periods(total_laps: int, rng: np.random.Generator,
                    rate: float = NEUTRAL_PER_RACE) -> set[int]:
    laps: set[int] = set()
    for _ in range(rng.poisson(rate)):
        start = int(rng.integers(2, total_laps - 2))
        length = int(rng.choice(NEUTRAL_LENGTHS))
        laps.update(range(start, min(start + length, total_laps)))
    return laps


def run_plan(m: TyreModel, plan: Plan, total_laps: int, green_loss: float, neutral_loss: float,
             neutral: set[int], pace: float, min_stint: int = 5) -> tuple[float, bool]:
    """Race time of `plan` in one draw, pitting early under a safety car when it pays."""
    seq = [s.compound for s in plan.stints]
    pits = list(plan.pit_laps)
    reacted = False
    for k in range(len(pits)):
        earlier = pits[k - 1] if k else 0
        # The first neutralised lap after the previous stop and before this one.
        lap = next((n for n in sorted(neutral) if earlier + min_stint <= n < pits[k]), None)
        if lap is None:
            continue
        moved = pits[:k] + [lap] + pits[k + 1:]
        cost = _stint_sum(m, seq, moved, total_laps, pace) - _stint_sum(m, seq, pits, total_laps, pace)
        if cost < green_loss - neutral_loss:
            pits, reacted = moved, True
    t = _stint_sum(m, seq, pits, total_laps, pace)
    t += sum(neutral_loss if p in neutral else green_loss for p in pits)
    return t, reacted


def simulate(laps: pd.DataFrame, plans: list[Plan], total_laps: int, green_losses: list[float],
             neutral_loss: float | None = None, draws: int = 500, models: int = 100,
             rate: float = NEUTRAL_PER_RACE, driver: int | None = None, seed: int = 0) -> list[Outcome]:
    """Run every candidate plan through the same `draws` races; most wins first."""
    rng = np.random.default_rng(seed)
    pool = bootstrap_models(laps, models, rng)
    green_losses = np.asarray(green_losses, dtype=float)
    times = np.empty((draws, len(plans)))
    react = np.zeros((draws, len(plans)), dtype=bool)
    for i in range(draws):
        m = pool[rng.integers(len(pool))]
        pace = m.reference_pace(driver) if driver in m.driver_pace else m.reference_pace()
        green = float(rng.choice(green_losses))
        neutral_cost = neutral_loss if neutral_loss is not None else green * NEUTRAL_PIT_SHARE
        neutral = neutral_periods(total_laps, rng, rate)
        for j, p in enumerate(plans):
            if any(s.compound not in m.compounds for s in p.stints):
                times[i, j] = np.inf
                continue
            times[i, j], react[i, j] = run_plan(m, p, total_laps, green, neutral_cost, neutral, pace)
    best = times.min(axis=1, keepdims=True)
    winner = times.argmin(axis=1)
    out = [Outcome(p, times[:, j] - best[:, 0], float(np.mean(winner == j)), float(react[:, j].mean()))
           for j, p in enumerate(plans)]
    return sorted(out, key=lambda o: (-o.wins, o.mean_gap))


def _stint_sum(m: TyreModel, seq: list[str], pits: list[int], total_laps: int, pace: float) -> float:
    bounds = (0, *pits, total_laps)
    stints = [Stint(c, bounds[i] + 1, bounds[i + 1]) for i, c in enumerate(seq)]
    return race_time(m, pace, stints, 0.0)
