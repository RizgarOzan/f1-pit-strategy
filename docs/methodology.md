# How the planner works

This is the long version of the "How it works" section in the README. Every step is in `src/f1strat/`, and each one is small enough to read in a sitting.

## 1. Data

Everything comes from the [OpenF1](https://openf1.org) API: lap times, tyre stints, pit stops, the classification and driver names. That's five requests per race. Responses are cached as JSON in `.cache/` (or wherever `F1STRAT_CACHE` points), so a race is downloaded once. `f1strat fetch 2023 2024 2025` fills the cache for whole seasons. The free API rate-limits bursts, so `fetch` backs off and retries on `429`.

Two feed quirks are handled in `openf1.py`:

- The pit endpoint is empty for some early-2023 races. The lap before a pit-out lap is an in-lap, so stops are rebuilt from the lap feed as well.
- The lap feed sometimes has one more lap than the race. The race length comes from the classification.

## 2. Clean laps

A lap is used for fitting only if it was raced flat out on dry tyres:

- not lap 1, not an in-lap, not an out-lap;
- not on a lap where the field's median time was more than 5% over its usual pace. That one rule catches safety cars, VSCs and red-flag restarts, with no flag data needed. The same list of laps feeds the Monte Carlo's safety car statistics;
- not more than 5% off the usual pace on its own (a spin, a puncture).

## 3. Lap time model

```
lap_time = driver_pace + fuel × lap_number + compound_offset + wear[compound] × tyre_age
```

One least-squares solve over the whole field (`model.py`). `driver_pace` takes in the car and driver, so the tyre and fuel terms are shared by everyone. Fuel and wear can only be told apart because different cars are on the same compound at different points in the race. The test suite builds a race where that holds and checks that the fit gets the true numbers back.

Wear is a straight line on purpose. A quadratic fit did no better on seen stints and got worse on tyres older than the ones it trained on, and judging a longer stint is the main thing a strategy search asks of the model. Because a line can't see a cliff, no planned stint may run more than 5 laps past the longest clean stint the data has on that compound.

## 4. Pit loss

For every green-flag stop: in-lap plus out-lap, minus two laps the model predicts for that driver on those tyres. Stops where the lap before wasn't clean, or where the in-lap or out-lap was neutralised, are left out. The planning number is the 40th percentile, so a stuck wheel nut doesn't set the plan. The whole list of stops goes to the Monte Carlo.

A stop under a safety car or VSC is measured differently: in-lap plus out-lap minus the field's median on those two laps, since the cars that stayed out were slow too.

## 5. Search

Under a linear model a stint's time has a closed form (a sum of an arithmetic series), so every one- and two-stop plan is scored exactly, not simulated lap by lap. A 70-lap race has about 40,000 plans and the search takes under a third of a second. Plans must use two dry compounds and every stint must be at least 5 laps. The fuel term doesn't depend on the compound, so medium-hard and hard-medium cost the same. The search keeps one per tyre set and lists the softest compound first.

The pit window for a stop is the range of laps it can move to, with the other stop held, while staying within 1 s of the plan's best time.

## 6. Monte Carlo

The search answers "what's fastest if the model is exactly right and nothing happens". `simulate.py` asks how often that answer holds when neither is true. Each simulated race:

1. picks a tyre model refit on a bootstrap resample of whole stints (laps of one stint stay together, so a compound with only a few stints gets a wide spread of wear rates);
2. draws a green-flag pit loss from the stops measured in that race;
3. draws safety car and VSC periods: a Poisson number per race at the rate measured over the backtest seasons, each 2 to 6 laps, starting anywhere after lap 1;
4. runs each candidate plan. When a neutralised lap comes up after the previous stop, the car pits then if moving the stop costs less in tyre time than the cheaper stop saves. Otherwise it stays on plan.

Out come three numbers per plan: the share of simulated races where it was fastest, the time it lost to the fastest plan of each race, and how often a safety car moved one of its stops. Lap-to-lap noise is left out because it hits every plan the same way.

## 7. Backtest

`backtest.py` runs all of the above over every race in the cache and compares it with what the field actually did. The field's strategy is reduced to two numbers: the most common stop count among classified finishers who stayed on dry tyres, and the median lap of their first stop. Tyre changes in the first or last three laps don't count (lap-1 red flags, punctures, fastest-lap stops, and one-lap stints the feed invents behind a final safety car).

Five predictors are scored against that:

- **search, same race**: fitted on the race it's predicting. In-sample, so an upper bound;
- **Monte Carlo, same race**: the plan that won most simulated races;
- **search, last year's race**: fitted on the same circuit one season earlier. The only honest pre-race test, since it never sees the race it predicts;
- **copy last year's field**: the baseline the forecast has to beat;
- **always one stop at half distance**: the naive floor.

Results and the known misses are in [validation.md](validation.md).
