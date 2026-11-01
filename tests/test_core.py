import numpy as np
import pandas as pd
import pytest

from f1strat.model import TyreModel
from f1strat.strategy import Stint, pit_window, race_time, search, stint_time

TRUE = {"fuel": -0.06, "offset": {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8},
        "deg": {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.03}}


def synthetic_laps(seed=0, noise=0.15):
    rng = np.random.default_rng(seed)
    rows = []
    # Fuel is only separable from wear because cars run the same compound at
    # different points of the race, so the drivers here split strategies.
    orders = [("SOFT", "MEDIUM", "HARD"), ("HARD", "MEDIUM", "SOFT"), ("MEDIUM", "HARD", "SOFT"), ("HARD", "SOFT", "MEDIUM")]
    for (drv, pace), order in zip([(1, 90.0), (16, 90.3), (44, 90.6), (63, 90.9)], orders):
        for compound, (first, last) in zip(order, [(2, 18), (19, 37), (38, 55)]):
            for lap in range(first, last + 1):
                age = lap - first
                t = (pace + TRUE["fuel"] * lap + TRUE["offset"][compound]
                     + TRUE["deg"][compound] * age + rng.normal(0, noise))
                rows.append((drv, lap, t, compound, age, compound))
    return pd.DataFrame(rows, columns=["driver_number", "lap_number", "lap_duration",
                                       "compound", "tyre_age", "stint"])


def known_model(**over):
    kw = dict(compounds=["SOFT", "MEDIUM", "HARD"], driver_pace={1: 90.0}, fuel=TRUE["fuel"],
              offset=TRUE["offset"], deg=TRUE["deg"], max_age={"SOFT": 40, "MEDIUM": 40, "HARD": 60})
    kw.update(over)
    return TyreModel(**kw)


def test_fit_recovers_known_parameters():
    m = TyreModel.fit(synthetic_laps())
    assert m.fuel == pytest.approx(TRUE["fuel"], abs=0.01)
    for c in m.compounds:
        assert m.deg[c] == pytest.approx(TRUE["deg"][c], abs=0.01)
        assert m.offset[c] == pytest.approx(TRUE["offset"][c], abs=0.15)
    assert m.driver_pace[16] - m.driver_pace[1] == pytest.approx(0.3, abs=0.05)


def test_closed_form_matches_lap_by_lap_sum():
    m = known_model()
    s = Stint("MEDIUM", first_lap=12, last_lap=30, start_age=3)
    laps = np.arange(s.first_lap, s.last_lap + 1)
    by_lap = m.lap_time("MEDIUM", s.start_age + laps - s.first_lap, laps, 90.0).sum()
    assert stint_time(m, 90.0, s) == pytest.approx(by_lap)


def test_every_plan_uses_two_compounds_and_covers_the_race():
    plans = search(known_model(), total_laps=50, pit_loss=22.0)
    assert plans
    for p in plans:
        assert len({s.compound for s in p.stints}) >= 2
        assert p.stints[0].first_lap == 1 and p.stints[-1].last_lap == 50
        assert all(a.last_lap + 1 == b.first_lap for a, b in zip(p.stints, p.stints[1:]))


def test_cheap_stops_and_heavy_wear_mean_more_stops():
    cheap = search(known_model(), total_laps=60, pit_loss=5.0)[0]
    dear = search(known_model(), total_laps=60, pit_loss=40.0)[0]
    assert cheap.stops == 2 and dear.stops == 1


def test_stints_never_run_far_past_what_the_data_saw():
    m = known_model(max_age={"SOFT": 10, "MEDIUM": 15, "HARD": 25})
    for p in search(m, total_laps=50, pit_loss=22.0, stretch=5):
        assert all(s.laps <= m.max_age[s.compound] + 5 for s in p.stints)


def test_window_contains_the_optimal_lap_and_best_time_is_consistent():
    m = known_model()
    best = search(m, total_laps=50, pit_loss=22.0)[0]
    assert best.time == pytest.approx(race_time(m, 90.0, best.stints, 22.0))
    for (lo, hi), lap in zip(pit_window(m, best, 50, 22.0), best.pit_laps):
        assert lo <= lap <= hi
