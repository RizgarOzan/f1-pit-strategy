"""Load one race from the OpenF1 API into tidy per-lap rows.

Responses are cached as JSON under ``.cache/`` so a race is downloaded once and
every later run works offline.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

API = "https://api.openf1.org/v1"
CACHE = Path(".cache")
DRY = ("SOFT", "MEDIUM", "HARD")


def fetch(endpoint: str, **params) -> list[dict]:
    query = urllib.parse.urlencode(params)
    path = CACHE / f"{endpoint}_{query.replace('&', '_').replace('=', '-')}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{API}/{endpoint}?{query}", timeout=120) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:  # OpenF1 answers 404 for "no rows"
                data = []
                break
            if e.code != 429 or attempt == 3:
                raise
        except OSError:  # TLS resets and read timeouts happen on this API
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


@dataclass
class Race:
    name: str
    total_laps: int
    laps: pd.DataFrame       # every lap with compound, tyre age and a `clean` flag
    pit_lane: pd.Series      # pit lane time per stop, seconds
    drivers: dict[int, str]  # number -> acronym
    finish_order: list[int]  # classified finishers, winner first


def find_race(year: int, location: str) -> dict:
    sessions = fetch("sessions", year=year, session_name="Race")
    key = location.lower()
    for s in sessions:
        names = (s["location"], s["circuit_short_name"], s["country_name"])
        if any(key == n.lower() for n in names):
            return s
    known = ", ".join(sorted({s["circuit_short_name"] for s in sessions}))
    raise SystemExit(f"No {year} race at '{location}'. Known circuits: {known}")


def load_race(year: int, location: str) -> Race:
    s = find_race(year, location)
    key = s["session_key"]
    laps = pd.DataFrame(fetch("laps", session_key=key))
    stints = pd.DataFrame(fetch("stints", session_key=key))
    pits = pd.DataFrame(fetch("pit", session_key=key))
    results = pd.DataFrame(fetch("session_result", session_key=key))
    drivers = {d["driver_number"]: d["name_acronym"] for d in fetch("drivers", session_key=key)}

    laps = laps[["driver_number", "lap_number", "lap_duration", "is_pit_out_lap"]].copy()
    laps = _attach_tyres(laps, stints)

    pit_in = set(zip(pits["driver_number"], pits["lap_number"])) if len(pits) else set()
    laps["pit_in"] = [(d, n) in pit_in for d, n in zip(laps.driver_number, laps.lap_number)]
    laps["clean"] = _clean_mask(laps)

    finished = results[results["number_of_laps"] == results["number_of_laps"].max()]
    order = finished.sort_values("position")["driver_number"].tolist()
    lane = pits["lane_duration"] if "lane_duration" in pits else pits.get("pit_duration")
    return Race(
        name=f"{year} {s['circuit_short_name']}",
        # The laps feed sometimes carries a phantom extra lap; the classification doesn't.
        total_laps=int(results["number_of_laps"].max()),
        laps=laps,
        pit_lane=pd.to_numeric(lane, errors="coerce").dropna(),
        drivers=drivers,
        finish_order=order,
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

    Drops lap 1, in/out laps and anything more than 5% off the field's median
    lap, which catches safety cars, VSC and incidents without needing flag data.
    """
    ok = (
        laps["lap_duration"].notna()
        & (laps["lap_number"] > 1)
        & ~laps["is_pit_out_lap"].fillna(False).astype(bool)
        & ~laps["pit_in"]
        & laps["compound"].isin(DRY)
    )
    field_median = laps.loc[ok].groupby("lap_number")["lap_duration"].median()
    typical = field_median.median()
    lap_ok = field_median[field_median < typical * 1.05].index
    ok &= laps["lap_number"].isin(lap_ok)
    ok &= laps["lap_duration"] < typical * 1.05
    return ok
