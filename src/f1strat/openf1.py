"""Load one race from the OpenF1 API into tidy per-lap rows.

Responses are cached as JSON under ``.cache/`` so a race is downloaded once and
every later run works offline.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

API = "https://api.openf1.org/v1"
CACHE = Path(os.environ.get("F1STRAT_CACHE", ".cache"))
DRY = ("SOFT", "MEDIUM", "HARD")


def fetch(endpoint: str, **params) -> list[dict]:
    query = urllib.parse.urlencode(params)
    path = CACHE / f"{endpoint}_{query.replace('&', '_').replace('=', '-')}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API}/{endpoint}?{query}", timeout=120) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenF1 answers 404 for "no rows"
                data = []
                break
            if e.code != 429 or attempt == 5:
                raise
        except OSError:  # TLS resets and read timeouts happen on this API
            if attempt == 5:
                raise
        time.sleep(3 * 2 ** attempt)  # the free tier rate-limits bursts
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


@dataclass
class Race:
    name: str
    total_laps: int
    laps: pd.DataFrame       # every lap with compound, tyre age and a `clean` flag
    drivers: dict[int, str]  # number -> acronym
    finish_order: list[int]  # classified finishers, winner first
    circuit: str = ""
    year: int = 0

    @property
    def neutralised(self) -> list[int]:
        return neutralised_laps(self.laps)


ENDPOINTS = ("laps", "stints", "pit", "session_result", "drivers")


def season(year: int) -> list[dict]:
    """Grand Prix races of a season that have already been run, in calendar order."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    races = [s for s in fetch("sessions", year=year, session_name="Race") if s["date_end"] < now]
    return sorted(races, key=lambda s: s["date_start"])


def download(year: int) -> int:
    """Fill the cache with every finished race of a season. Returns the race count."""
    races = season(year)
    for s in races:
        for endpoint in ENDPOINTS:
            fetch(endpoint, session_key=s["session_key"])
    return len(races)


def find_race(year: int, location: str) -> dict:
    sessions = season(year)
    key = location.lower()
    for s in sessions:
        names = (s["location"], s["circuit_short_name"], s["country_name"])
        if any(key == n.lower() for n in names):
            return s
    known = ", ".join(sorted({s["circuit_short_name"] for s in sessions}))
    raise SystemExit(f"No {year} race at '{location}'. Known circuits: {known}")


def load_race(year: int, location: str) -> Race:
    return load_session(find_race(year, location))


def load_session(s: dict) -> Race:
    key, year = s["session_key"], s["year"]
    laps = pd.DataFrame(fetch("laps", session_key=key))
    if laps.empty:
        raise ValueError("no lap data (race not held)")
    stints = pd.DataFrame(fetch("stints", session_key=key))
    pits = pd.DataFrame(fetch("pit", session_key=key))
    results = pd.DataFrame(fetch("session_result", session_key=key))
    drivers = {d["driver_number"]: d["name_acronym"] for d in fetch("drivers", session_key=key)}

    laps = laps[["driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]].copy()
    laps = _attach_tyres(laps, stints)

    pit_in = set(zip(pits["driver_number"], pits["lap_number"])) if len(pits) else set()
    # The pit feed is empty for some early-2023 races; the lap before an out-lap is an in-lap.
    out = laps[laps["is_pit_out_lap"].fillna(False).astype(bool) & (laps["lap_number"] > 1)]
    pit_in |= set(zip(out["driver_number"], out["lap_number"] - 1))
    laps["pit_in"] = [(d, n) in pit_in for d, n in zip(laps.driver_number, laps.lap_number)]
    laps["clean"] = _clean_mask(laps)

    finished = results[results["number_of_laps"] == results["number_of_laps"].max()]
    order = finished.sort_values("position")["driver_number"].tolist()
    return Race(
        name=f"{year} {s['circuit_short_name']}",
        # The laps feed sometimes carries a phantom extra lap; the classification doesn't.
        total_laps=int(results["number_of_laps"].max()),
        laps=laps,
        drivers=drivers,
        finish_order=order,
        circuit=s["circuit_short_name"],
        year=year,
    )


def _attach_tyres(laps: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    laps["compound"] = None
    laps["tyre_age"] = pd.NA
    laps["stint"] = pd.NA
    for st in stints.itertuples():
        if pd.isna(st.lap_start) or pd.isna(st.lap_end):
            continue
        m = (laps.driver_number == st.driver_number) & laps.lap_number.between(st.lap_start, st.lap_end)
        laps.loc[m, "compound"] = st.compound
        laps.loc[m, "stint"] = st.stint_number
        age0 = 0 if pd.isna(st.tyre_age_at_start) else st.tyre_age_at_start
        laps.loc[m, "tyre_age"] = age0 + laps.loc[m, "lap_number"] - st.lap_start
    laps["tyre_age"] = pd.to_numeric(laps["tyre_age"])
    return laps


def _clean_mask(laps: pd.DataFrame) -> pd.Series:
    """Green-flag racing laps only.

    Drops lap 1, in/out laps, neutralised laps and single laps more than 5% off
    the field's usual pace (a spin, a slow puncture lap).
    """
    ok = _racing(laps)
    typical = laps.loc[ok].groupby("lap_number")["lap_duration"].median().median()
    ok &= ~laps["lap_number"].isin(neutralised_laps(laps))
    ok &= laps["lap_duration"] < typical * 1.05
    return ok


def neutralised_laps(laps: pd.DataFrame) -> list[int]:
    """Laps where the field's median time was more than 5% over its usual pace.

    That catches safety cars, VSCs and red-flag restarts without needing flag
    data, which OpenF1 doesn't have for every season.
    """
    field = laps.loc[_racing(laps)].groupby("lap_number")["lap_duration"].median()
    return sorted(int(n) for n in field[field > field.median() * 1.05].index)


def _racing(laps: pd.DataFrame) -> pd.Series:
    return (
        laps["lap_duration"].notna()
        & (laps["lap_number"] > 1)
        & ~laps["is_pit_out_lap"].fillna(False).astype(bool)
        & ~laps["pit_in"]
        & laps["compound"].isin(DRY)
    )
