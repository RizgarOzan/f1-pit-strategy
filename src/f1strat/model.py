"""Lap time model: car pace + fuel burn + compound offset + tyre degradation.

    lap_time = driver_pace + fuel * lap_number
               + compound_offset[c] + deg[c] * tyre_age

Every term means something a strategist would recognise. Degradation is kept
linear on purpose: a quadratic term fit stints slightly better but blew up on
older tyres than it trained on, and judging a long stint is exactly what a
strategy search asks of the model (numbers in the README).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


@dataclass
class TyreModel:
    compounds: list[str]
    driver_pace: dict[int, float]
    fuel: float
    offset: dict[str, float]
    deg: dict[str, float]
    max_age: dict[str, int]  # longest clean stint seen per compound

    @classmethod
    def fit(cls, laps: pd.DataFrame, min_laps: int = 30) -> "TyreModel":
        counts = laps["compound"].value_counts()
        compounds = [c for c in ("SOFT", "MEDIUM", "HARD") if counts.get(c, 0) >= min_laps]
        laps = laps[laps["compound"].isin(compounds)]
        X, drivers = _design(laps, compounds)
        # A tiny ridge keeps the solve stable when a compound has few laps; it
        # barely moves the coefficients (seconds-scale targets).
        reg = Ridge(alpha=1e-3, fit_intercept=False).fit(X, laps["lap_duration"].to_numpy())
        w = reg.coef_
        nd, nc = len(drivers), len(compounds)
        pace = {int(d): float(v) for d, v in zip(drivers, w[:nd])}
        fuel = w[nd]
        off = {c: 0.0 for c in compounds}
        off.update({c: float(v) for c, v in zip(compounds[1:], w[nd + 1 : nd + nc])})
        base = nd + nc
        deg = {c: float(v) for c, v in zip(compounds, w[base : base + nc])}
        max_age = laps.groupby("compound")["tyre_age"].max().astype(int).to_dict()
        return cls(compounds, pace, float(fuel), off, deg, max_age)

    def reference_pace(self, driver: int | None = None) -> float:
        if driver is not None:
            return self.driver_pace[driver]
        return float(np.median(list(self.driver_pace.values())))

    def lap_time(self, compound: str, age, lap, pace: float):
        age = np.asarray(age, dtype=float)
        return (pace + self.fuel * np.asarray(lap, dtype=float) + self.offset[compound]
                + self.deg[compound] * age)

    def predict(self, laps: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(laps))
        for i, r in enumerate(laps.itertuples()):
            pace = self.driver_pace.get(r.driver_number, self.reference_pace())
            out[i] = self.lap_time(r.compound, r.tyre_age, r.lap_number, pace)
        return out


def _design(laps: pd.DataFrame, compounds: list[str]):
    drivers = sorted(laps["driver_number"].unique())
    n = len(laps)
    d_idx = {d: i for i, d in enumerate(drivers)}
    nd, nc = len(drivers), len(compounds)
    X = np.zeros((n, nd + 1 + (nc - 1) + nc))
    rows = np.arange(n)
    X[rows, [d_idx[d] for d in laps["driver_number"]]] = 1
    X[:, nd] = laps["lap_number"].to_numpy()
    c_idx = np.array([compounds.index(c) for c in laps["compound"]])
    age = laps["tyre_age"].to_numpy(dtype=float)
    for k in range(1, nc):
        X[c_idx == k, nd + k] = 1
    base = nd + nc
    X[rows, base + c_idx] = age
    return X, drivers


def pit_loss(race_laps: pd.DataFrame, model: TyreModel) -> float | None:
    """Time a stop costs vs staying out: (in-lap + out-lap) minus two normal laps.

    Measured from the race itself, so it already includes this track's pit lane
    length and speed limit. Stops under safety car are skipped by requiring
    the out-lap's neighbours to be clean.
    """
    laps = race_laps.set_index(["driver_number", "lap_number"]).sort_index()
    losses = []
    for (drv, n), row in laps[laps["pit_in"]].iterrows():
        nxt = laps.loc[(drv, n + 1)] if (drv, n + 1) in laps.index else None
        before = laps.loc[(drv, n - 1)] if (drv, n - 1) in laps.index else None
        if nxt is None or before is None or not before["clean"]:
            continue
        if nxt["compound"] not in model.compounds or row["compound"] not in model.compounds:
            continue
        if pd.isna(row["lap_duration"]) or pd.isna(nxt["lap_duration"]):
            continue
        pace = model.driver_pace.get(drv)
        if pace is None:
            continue
        normal = (model.lap_time(row["compound"], row["tyre_age"], n, pace)
                  + model.lap_time(nxt["compound"], nxt["tyre_age"], n + 1, pace))
        losses.append(row["lap_duration"] + nxt["lap_duration"] - normal)
    if len(losses) < 3:
        return None
    # Slow stops (stuck wheel nut, unsafe release hold) are real but not the plan.
    return float(np.percentile(losses, 40))


def evaluate(laps: pd.DataFrame, folds: int = 5) -> dict[str, float]:
    """Compare the physical model with gradient boosting on two tasks.

    interpolation: hold out whole stints (GroupKFold by driver+stint).
    extrapolation: train on tyre age <= 60th percentile, test on older tyres,
                   which is what judging a long first stint asks of a model.
    """
    laps = laps.reset_index(drop=True)
    compounds = TyreModel.fit(laps).compounds
    laps = laps[laps["compound"].isin(compounds)].reset_index(drop=True)
    groups = laps["driver_number"].astype(str) + "-" + laps["stint"].astype(str)
    res: dict[str, list[float]] = {"phys": [], "gbm": []}
    for tr, te in GroupKFold(n_splits=folds).split(laps, groups=groups):
        phys, gbm = _split_scores(laps.iloc[tr], laps.iloc[te])
        res["phys"].append(phys)
        res["gbm"].append(gbm)
    cut = laps["tyre_age"].quantile(0.6)
    extrap_phys, extrap_gbm = _split_scores(laps[laps["tyre_age"] <= cut], laps[laps["tyre_age"] > cut])
    return {
        "interp_phys": float(np.mean(res["phys"])),
        "interp_gbm": float(np.mean(res["gbm"])),
        "extrap_phys": extrap_phys,
        "extrap_gbm": extrap_gbm,
        "age_cut": float(cut),
    }


def _split_scores(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    phys = TyreModel.fit(train)
    # Score only what both models could have learned: known drivers and compounds.
    test = test[test["driver_number"].isin(phys.driver_pace) & test["compound"].isin(phys.compounds)]
    return (float(mean_absolute_error(test["lap_duration"], phys.predict(test))),
            float(mean_absolute_error(test["lap_duration"], _gbm(train).predict(_gbm_x(test)))))


def _gbm_x(laps: pd.DataFrame) -> np.ndarray:
    comp = laps["compound"].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2})
    return np.column_stack([laps["driver_number"], laps["lap_number"], comp, laps["tyre_age"]])


def _gbm(laps: pd.DataFrame) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      categorical_features=[0, 2], random_state=0)
    return m.fit(_gbm_x(laps), laps["lap_duration"])
