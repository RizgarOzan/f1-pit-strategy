"""f1strat plan 2024 Monza | simulate 2024 Monza | backtest 2023 2024 2025 | fetch 2025"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import TyreModel, evaluate, neutral_pit_losses, pit_loss, pit_losses
from .openf1 import Race, download, load_race
from .strategy import Plan, actual_stints, pit_window, race_time, search

DEFAULT_PIT_LOSS = 22.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="f1strat", description="Pit strategy planning from OpenF1 timing data.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def race_args(p):
        p.add_argument("year", type=int)
        p.add_argument("race", help="location, circuit or country, e.g. Monza, Sakhir, Spain")
        p.add_argument("--driver", help="plan around this driver's pace (acronym, e.g. VER); default: field median")
        p.add_argument("--pit-loss", type=float, help="override the pit loss measured from the race, seconds")
        p.add_argument("--top", type=int, default=5, help="plans to list")

    p = sub.add_parser("plan", help="fit the tyre model and rank one- and two-stop plans")
    race_args(p)
    p.add_argument("--evaluate", action="store_true", help="also cross-validate the model against gradient boosting")
    p.add_argument("--plot", help="save a chart to this PNG path")

    p = sub.add_parser("simulate", help="Monte Carlo the best plans through safety cars and model uncertainty")
    race_args(p)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", help="save a chart to this PNG path")

    p = sub.add_parser("backtest", help="compare the planner with what the field did, race by race")
    p.add_argument("years", type=int, nargs="+")
    p.add_argument("--out", default="results/backtest.csv")
    p.add_argument("--draws", type=int, default=300)
    p.add_argument("--no-lap-model", action="store_true", help="skip the slower lap-time model comparison")

    p = sub.add_parser("fetch", help="download every finished race of a season into the cache")
    p.add_argument("years", type=int, nargs="+")

    a = ap.parse_args(argv)
    {"plan": cmd_plan, "simulate": cmd_simulate, "backtest": cmd_backtest, "fetch": cmd_fetch}[a.cmd](a)


def setup(a):
    """Load the race, fit the model and settle pace and pit loss, printing what was used."""
    race = load_race(a.year, a.race)
    clean = race.laps[race.laps["clean"]]
    model = TyreModel.fit(clean)
    driver, who = None, "field median pace"
    if a.driver:
        driver = {v: k for k, v in race.drivers.items()}.get(a.driver.upper())
        if driver not in model.driver_pace:
            raise SystemExit(f"No clean laps for driver '{a.driver}'.")
        who = f"{a.driver.upper()}'s pace"
    pace = model.reference_pace(driver)

    measured = pit_loss(race.laps, model)
    loss = a.pit_loss or measured or DEFAULT_PIT_LOSS
    src = "given" if a.pit_loss else ("measured from this race" if measured else "default, too few clean stops")

    print(f"\n{race.name}: {race.total_laps} laps, {len(clean)} clean laps, planning on {who}")
    print(f"Pit loss {loss:.1f} s ({src})")
    for c in model.compounds:
        print(f"  {c:<6}  {model.offset[c]:+.2f} s vs {model.compounds[0].lower()},"
              f"  {model.deg[c]:+.3f} s/lap wear,  longest clean stint {model.max_age[c]} laps")
    print(f"  fuel    {model.fuel:+.3f} s/lap as the car gets lighter")
    return race, clean, model, driver, pace, loss


def cmd_plan(a) -> None:
    race, clean, model, _, pace, loss = setup(a)
    if a.evaluate:
        e = evaluate(clean)
        print("\nLap-time error (MAE), this model vs gradient boosting:")
        older = f"tyres older than {e['age_cut']:.0f} laps"
        print(f"  {'held-out stints':<26}{e['interp_phys']:.3f} s  vs  {e['interp_gbm']:.3f} s")
        print(f"  {older:<26}{e['extrap_phys']:.3f} s  vs  {e['extrap_gbm']:.3f} s")

    plans = search(model, race.total_laps, loss, pace)
    best = plans[0].time
    print(f"\n{'#':>2}  {'plan':<8} {'stops':>5}  {'pit lap':<10} {'window (within 1 s)':<20} {'gap':>8}")
    for i, p in enumerate(plans[: a.top], 1):
        win = pit_window(model, p, race.total_laps, loss, pace)
        laps = ", ".join(map(str, p.pit_laps))
        wins = ", ".join(str(x) if x == y else f"{x}-{y}" for x, y in win)
        gap = "fastest" if i == 1 else f"+{p.time - best:.1f} s"
        print(f"{i:>2}  {p.label:<8} {p.stops:>5}  {laps:<10} {wins:<20} {gap:>8}")

    rows = [(p.label, p, p.time - best) for p in plans[: min(a.top, 4)]]
    winner = winner_row(race, model, pace, loss, best)
    if winner:
        rows.append(winner)

    if a.plot:
        from .plot import figure
        figure(race.name, clean, model, rows, race.total_laps, a.plot)
        print(f"\nChart saved to {a.plot}")


def winner_row(race: Race, model: TyreModel, pace: float, loss: float, best: float):
    if not race.finish_order:
        return None
    winner = race.finish_order[0]
    stints = actual_stints(race.laps, winner, race.total_laps)
    name = race.drivers.get(winner, str(winner))
    if not stints or any(s.compound not in model.compounds for s in stints):
        print(f"\nWinner {name} used a compound with too few clean laps to model; skipping the comparison.")
        return None
    t = race_time(model, pace, stints, loss)
    ran = " ".join(f"{s.compound[0]}{s.first_lap}-{s.last_lap}" for s in stints)
    print(f"\nWinner {name} ran {ran}: {t - best:+.1f} s vs the best plan on the same pace.")
    print("  (The model knows nothing about safety cars, traffic or undercuts; a gap here is"
          " where the race itself decided.)")
    return f"{name} (actual)", Plan(tuple(stints), t), t - best


def cmd_simulate(a) -> None:
    from .simulate import simulate

    race, clean, model, driver, pace, loss = setup(a)
    plans = search(model, race.total_laps, loss, pace)[: a.top]
    green = [loss] if a.pit_loss else (pit_losses(race.laps, model) or [loss])
    under = neutral_pit_losses(race.laps, race.neutralised)
    neutral = float(np.median(under)) if len(under) >= 3 else None
    out = simulate(clean, plans, race.total_laps, green, neutral, draws=a.draws, driver=driver, seed=a.seed)
    print(f"\n{a.draws} simulated races: refit tyre model, drawn pit loss, random safety cars")
    print(f"{'plan':<8} {'pit lap':<10} {'wins':>6} {'mean gap':>9} {'90% worst':>10} {'pitted under SC':>16}")
    for o in out:
        laps = ", ".join(map(str, o.plan.pit_laps))
        print(f"{o.plan.label:<8} {laps:<10} {o.wins:>6.0%} {o.mean_gap:>8.1f}s"
              f" {np.percentile(o.gaps, 90):>9.1f}s {o.reacted:>15.0%}")
    if a.plot:
        from .plot import monte_carlo_figure
        monte_carlo_figure(race.name, out, a.plot)
        print(f"\nChart saved to {a.plot}")


def cmd_backtest(a) -> None:
    from .backtest import run, summary

    df = run(a.years, a.draws, lap_model=not a.no_lap_model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.3f")
    s = summary(df)
    s.to_csv(Path(a.out).with_name("summary.csv"), index=False, float_format="%.3f")
    from .plot import backtest_figure
    backtest_figure(df, s, str(Path(a.out).with_name("backtest.png")))
    ok = df[df["skip"] == ""]
    print(f"\n{len(ok)} of {len(df)} races checked, rows in {a.out}\n")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nSafety car / VSC periods per race: {ok['neutral_periods'].mean():.2f},"
          f" {ok['neutral_laps'].sum() / max(ok['neutral_periods'].sum(), 1):.1f} laps each on average")
    share = (ok["neutral_loss"] / ok["green_loss"]).dropna()
    print(f"A stop under neutralisation cost {share.median():.0%} of a green one (median of {len(share)} races)")
    if not a.no_lap_model:
        cols = ["mae_stints", "mae_stints_gbm", "mae_old", "mae_old_gbm"]
        m = ok[cols].mean()
        print(f"Lap-time MAE, held-out stints: {m.iloc[0]:.3f} s vs GBM {m.iloc[1]:.3f} s;"
              f" older tyres: {m.iloc[2]:.3f} s vs GBM {m.iloc[3]:.3f} s"
              f" (model better in {(ok.mae_old < ok.mae_old_gbm).mean():.0%} of races on older tyres)")


def cmd_fetch(a) -> None:
    for y in a.years:
        print(f"{y}: {download(y)} races cached")


if __name__ == "__main__":
    main()
