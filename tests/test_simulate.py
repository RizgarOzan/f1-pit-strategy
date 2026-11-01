import pytest
from test_core import known_model, synthetic_laps

from f1strat.backtest import periods
from f1strat.openf1 import neutralised_laps
from f1strat.simulate import run_plan, simulate
from f1strat.strategy import race_time, search


def test_green_race_costs_exactly_the_closed_form():
    m = known_model()
    plan = search(m, total_laps=50, pit_loss=22.0)[0]
    t, reacted = run_plan(m, plan, 50, green_loss=22.0, neutral_loss=12.0, neutral=set(), pace=90.0)
    assert t == pytest.approx(race_time(m, 90.0, plan.stints, 22.0))
    assert not reacted


def test_safety_car_inside_the_window_pulls_the_stop_forward():
    m = known_model()
    plan = search(m, total_laps=50, pit_loss=22.0)[0]
    stop = plan.pit_laps[0]
    neutral = {stop - 2, stop - 1}
    t, reacted = run_plan(m, plan, 50, green_loss=22.0, neutral_loss=10.0, neutral=neutral, pace=90.0)
    assert reacted
    assert t < plan.time  # a 12 s cheaper stop beats two laps of tyre life


def test_safety_car_too_early_is_ignored():
    m = known_model()
    plan = search(m, total_laps=50, pit_loss=22.0)[0]
    # Pitting on lap 6 would leave an impossible last stint, so the plan stays put.
    t, reacted = run_plan(m, plan, 50, green_loss=22.0, neutral_loss=21.5, neutral={6}, pace=90.0)
    assert not reacted
    assert t == pytest.approx(plan.time)


def test_without_safety_cars_the_search_winner_wins_most_draws():
    laps = synthetic_laps()
    plans = search(known_model(), total_laps=55, pit_loss=22.0)[:4]
    out = simulate(laps, plans, 55, green_losses=[22.0], draws=200, models=40, rate=0.0)
    assert out[0].plan == plans[0]
    assert sum(o.wins for o in out) == pytest.approx(1.0)
    assert all(o.reacted == 0 for o in out)


def test_neutralised_laps_are_found_from_lap_times_alone():
    laps = synthetic_laps()
    laps["is_pit_out_lap"] = False
    laps["pit_in"] = False
    slow = laps["lap_number"].isin([20, 21, 22])
    laps.loc[slow, "lap_duration"] *= 1.3
    assert neutralised_laps(laps) == [20, 21, 22]


def test_periods_split_runs_of_laps():
    assert periods([10, 11, 12, 30, 31, 45]) == [3, 2, 1]
    assert periods([]) == []


def test_a_stint_split_by_the_feed_is_joined_back():
    import pandas as pd

    from f1strat.strategy import actual_stints

    rows = [(n, 1, "HARD", n - 1) for n in range(1, 21)]          # one set, laps 1-20
    rows += [(n, 2, "HARD", n - 1) for n in range(21, 24)]        # feed opens a new stint, same set
    rows += [(n, 3, "MEDIUM", n - 24) for n in range(24, 41)]     # real stop
    laps = pd.DataFrame(rows, columns=["lap_number", "stint", "compound", "tyre_age"]).assign(driver_number=1)
    st = actual_stints(laps, 1, 40)
    assert [(s.compound, s.first_lap, s.last_lap) for s in st] == [("HARD", 1, 23), ("MEDIUM", 24, 40)]
