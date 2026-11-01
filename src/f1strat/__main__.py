"""python -m f1strat 2024 Monza [--driver LEC] [--evaluate] [--plot out.png]"""
from __future__ import annotations

import argparse

from .model import TyreModel, evaluate, pit_loss
from .openf1 import load_race
from .strategy import Plan, actual_stints, pit_window, race_time, search

DEFAULT_PIT_LOSS = 22.0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="f1strat", description="Find the fastest pit strategy for an F1 race.")
    ap.add_argument("year", type=int)
    ap.add_argument("race", help="location, circuit or country, e.g. Monza, Sakhir, Spain")
    ap.add_argument("--driver", help="plan around this driver's pace (acronym, e.g. VER); default: field median")
    ap.add_argument("--pit-loss", type=float, help="override the pit loss measured from the race, seconds")
    ap.add_argument("--top", type=int, default=5, help="plans to list")
    ap.add_argument("--evaluate", action="store_true", help="also cross-validate the model against gradient boosting")
    ap.add_argument("--plot", help="save a chart to this PNG path")
    a = ap.parse_args(argv)

    race = load_race(a.year, a.race)
    clean = race.laps[race.laps["clean"]]
    model = TyreModel.fit(clean)

    pace, who = model.reference_pace(), "field median pace"
    if a.driver:
        num = {v: k for k, v in race.drivers.items()}.get(a.driver.upper())
        if num not in model.driver_pace:
            raise SystemExit(f"No clean laps for driver '{a.driver}'.")
        pace, who = model.reference_pace(num), f"{a.driver.upper()}'s pace"

    measured = pit_loss(race.laps, model)
    loss = a.pit_loss or measured or DEFAULT_PIT_LOSS
    src = "given" if a.pit_loss else ("measured from this race" if measured else "default, too few clean stops")

    print(f"\n{race.name}: {race.total_laps} laps, {len(clean)} clean laps, planning on {who}")
    print(f"Pit loss {loss:.1f} s ({src})")
    for c in model.compounds:
        print(f"  {c:<6}  {model.offset[c]:+.2f} s vs {model.compounds[0].lower()},"
              f"  {model.deg[c]:+.3f} s/lap wear,  longest clean stint {model.max_age[c]} laps")
    print(f"  fuel    {model.fuel:+.3f} s/lap as the car gets lighter")

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
    winner = race.finish_order[0] if race.finish_order else None
    if winner is not None:
        stints = actual_stints(race.laps, winner, race.total_laps)
        name = race.drivers.get(winner, str(winner))
        if stints and all(s.compound in model.compounds for s in stints):
            t = race_time(model, pace, stints, loss)
            ran = " ".join(f"{s.compound[0]}{s.first_lap}-{s.last_lap}" for s in stints)
            print(f"\nWinner {name} ran {ran}: {t - best:+.1f} s vs the best plan on the same pace.")
            print("  (The model knows nothing about safety cars, traffic or undercuts; a gap here is"
                  " where the race itself decided.)")
            rows.append((f"{name} (actual)", Plan(tuple(stints), t), t - best))
        else:
            print(f"\nWinner {name} used a compound with too few clean laps to model; skipping the comparison.")

    if a.plot:
        from .plot import figure
        figure(race.name, clean, model, rows, race.total_laps, a.plot)
        print(f"\nChart saved to {a.plot}")


if __name__ == "__main__":
    main()
