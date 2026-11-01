# Validation

Everything here comes from one run of

```bash
f1strat backtest 2023 2024 2025 2026
```

with the cache filled on 1 November 2026. Per-race rows are in [`results/backtest.csv`](../results/backtest.csv), the summary in [`results/summary.csv`](../results/summary.csv). How the field's strategy is read and what each predictor means is in [methodology.md](methodology.md#7-backtest).

## Which races

91 races were in the cache and 79 were checked. The other 12:

- not held, no lap data: Imola 2023, Sakhir and Jeddah 2026;
- fewer than 5 finishers stayed on dry tyres, so there's no dry strategy to compare with: Monaco and Zandvoort 2023; Montreal, Silverstone and Interlagos 2024; Melbourne, Silverstone and Spa 2025; Montreal 2026.

The forecast and copy-last-year predictors need the same circuit a season earlier, which leaves 48 races for them.

## Stop count and first stop

| predictor | races | stop count right | first stop, mean error (laps) | first stop within 3 laps |
|---|---|---|---|---|
| Monte Carlo, same race | 79 | 76% | 6.18 | 32% |
| search, same race | 79 | 73% | 6.16 | 32% |
| search, last year's race | 48 | 60% | 7.54 | 23% |
| copy last year's field | 48 | 60% | 7.59 | 30% |
| one stop at half distance | 79 | 51% | 10.54 | 17% |

The first-stop error is only meaningful next to the stop count: a one-stop plan pitting on lap 30 against a field that two-stopped from lap 18 counts as 12 laps off.

By season, stop count right (search / Monte Carlo):

| season | races | search | Monte Carlo |
|---|---|---|---|
| 2023 | 20 | 70% | 75% |
| 2024 | 21 | 81% | 81% |
| 2025 | 21 | 67% | 67% |
| 2026 | 17 | 76% | 82% |

On the 48 races with a prior year, the in-sample search got 36 right, the forecast 29 and copying last year 29. A season-old fit adds nothing over just repeating what the field did then.

## Where it goes wrong

The search missed the stop count in 21 races:

| model said | field did | races |
|---|---|---|
| 1 stop | 2 stops | 13 |
| 2 stops | 1 stop | 3 |
| 2 stops | 3 stops | 2 |
| 1 stop | 3 stops | 1 |
| 1 stop | no stop after lap 3 | 2 |

The last row is races where a lap-1 red flag gave everyone a free tyre change and nobody stopped again (Monaco 2024 is one). The model has no way of knowing that.

The one-stop lean and the timing go together. In the 58 races where the stop count was right, the model's first stop was 2.2 laps later than the field's on average, and later in 66% of them. A tyre model on an empty track has no reason to pit before the tyres are worth changing. Real teams pit earlier to get ahead of the car in front (the undercut) or to cover it. Two stops make that easier, so the field picks two where the tyre maths alone says one.

The search and the Monte Carlo disagreed in 6 races. The Monte Carlo was right in 4 (Singapore 2023, Abu Dhabi 2023, Hungary 2025, Hungary 2026) and the search in 2 (Mexico 2023, Monaco 2025).

## The winner's strategy

Replaying the winner's real stints through the model of that race gives the time they gave away against the best plan on the same pace. Over 75 races with a winner on modelled compounds, the median was 6.0 s, and 41% were within 5 s. That's not a measure of error: the winner usually traded a few seconds of tyre time for track position or a safety car, which is the part the model leaves out. Monza 2024 (Leclerc, +5.3 s) is the worked example in the README.

## Monte Carlo inputs

Measured over the same races and used as the simulator's defaults:

- 1.19 safety car or VSC periods per race, 2.8 laps each on average;
- a stop under neutralisation cost 66% of a green-flag stop (median over 40 races with at least 3 such stops);
- green-flag pit loss, 40th percentile per race: 17 to 31 s, median 22.

Across all races, the plan that won most simulated races won a median 68% of them. So for a typical race the "best" plan loses roughly one simulated race in three to another plan.

## Lap time model vs gradient boosting

Median MAE per race over the 79 races:

| test | linear model | GBM | linear better in |
|---|---|---|---|
| held-out stints | 0.49 s | 0.63 s | 89% of races |
| tyres older than the training laps | 0.55 s | 0.58 s | 54% of races |

The mean on older tyres is 1.00 s against 0.61 s, and one race causes it. In Miami 2025 the older-tyre split trains only on laps 25 and later, where tyre age and lap number rise together for most of the field, so fuel and wear can't be separated and the extrapolation is 31 s off. The fit on the full race, which is what plans use, doesn't have that problem.

## Feed quirks found along the way

- The early-2023 pit feed is empty. Stops are rebuilt from out-laps.
- Pit loss came out at 95-118 s in races where the field pitted under a safety car (Singapore 2023, Jeddah 2024). Stops with a neutralised in-lap or out-lap are now left out of the green number.
- Behind a late safety car the stints feed sometimes opens a new stint on the same set every lap (Montreal 2025 read as a 5-stop race). Consecutive stints on the same compound whose tyre age carries on are joined back together.
