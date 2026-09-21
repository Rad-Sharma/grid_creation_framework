"""Section V.B.1: cached nearby OSM-name inference for unnamed plants."""
import hashlib
import json
from pathlib import Path
import time
import warnings

from geopy.distance import geodesic
import requests

# Overpass rate-limits, and a region can have hundreds of unnamed plants: Taiwan
# has 447. Each failed lookup costs up to the request timeout, so once the
# endpoint has clearly stopped answering there is nothing to gain from asking
# again several hundred more times - every remaining answer is None either way,
# and the warning already says a manual lookup is needed. Give up instead of
# spending an hour proving it.
MAX_CONSECUTIVE_FAILURES = 5
# The region-wide query replaces all of them, so it earns a few retries.
REGION_QUERY_ATTEMPTS = 4
REGION_QUERY_BACKOFF_S = 5.0
_state = {"failures": 0, "announced": False}


def reset_failure_count():
    """Forget earlier failures; a fresh run or a new region starts clean."""
    _state.update(failures=0, announced=False)


def region_named_power_objects(bbox, cache_dir, session=requests):
    """Every named power plant, generator and substation in the bounding box.

    Sec. V.B.1's inference is per plant, but asking Overpass once per plant means
    one rate-limited round trip each: Taiwan has 337 unnamed plants, which took
    over an hour of nothing but waiting. One bbox query returns the same
    candidates, and the nearest-match ranking is then a local calculation.

    Returns a list of (latitude, longitude, name, osm_id), or None if the query
    fails, in which case the caller falls back to querying per plant.
    """
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    cache_dir = Path(cache_dir)
    key = hashlib.sha256(
        f"region,{min_lon:.5f},{min_lat:.5f},{max_lon:.5f},{max_lat:.5f}".encode()
    ).hexdigest()[:20]
    path = cache_dir / f"power_names_{key}.json"
    if path.exists():
        return _candidates(json.loads(path.read_text(encoding="utf-8")))
    query = (f'[out:json][timeout:180];'
             f'nwr({min_lat},{min_lon},{max_lat},{max_lon})'
             '["power"~"^(plant|generator|substation)$"]["name"];out center tags;')
    # Worth retrying, unlike the per-plant path: this is one request standing in
    # for hundreds, so a rate-limit refusal here costs the whole saving.
    failure = None
    for attempt in range(REGION_QUERY_ATTEMPTS):
        try:
            result = session.post("https://overpass-api.de/api/interpreter", data={"data": query},
                                  headers={"User-Agent": "synthetic-grid-research/1.0"}, timeout=240)
            result.raise_for_status()
            response = result.json()
            if "elements" not in response or "remark" in response:
                raise ValueError("Overpass query incomplete")
        except (requests.RequestException, ValueError) as exc:
            failure = exc
            if attempt + 1 < REGION_QUERY_ATTEMPTS:
                time.sleep(REGION_QUERY_BACKOFF_S * 2**attempt)
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response), encoding="utf-8")
        return _candidates(response)
    warnings.warn(f"Region-wide plant-name query failed {REGION_QUERY_ATTEMPTS} times, "
                  f"falling back to one query per plant: {failure}", stacklevel=2)
    return None


def _candidates(response):
    """Named power objects with usable coordinates, as (lat, lon, name, osm_id)."""
    found = []
    for element in response.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name or tags.get("power") not in {"plant", "generator", "substation"}:
            continue
        center = element.get("center", element)
        if "lat" not in center or "lon" not in center:
            continue
        found.append((center["lat"], center["lon"], name,
                      f"{element.get('type', '?')}/{element.get('id', '?')}"))
    return found


def nearest_named(latitude, longitude, candidates, radius_m=1000):
    """Closest named power object within the radius, as (bracketed name, osm id)."""
    within = []
    for lat, lon, name, osm_id in candidates:
        metres = geodesic((latitude, longitude), (lat, lon)).meters
        if metres <= radius_m:
            within.append((metres, osm_id, name))
    if not within:
        return None, None
    _, osm_id, name = min(within)
    return f"[{name}]", osm_id


def nearby_plant_name(latitude, longitude, cache_dir, radius_m=1000, session=requests):
    """Return (bracketed name, OSM reference), or leave the plant unnamed.

    The manuscript does not specify the search radius or ranking. We use the
    closest named plant/generator/substation within a configurable radius and
    record its ID. Inference never fills capacity or fuel from another plant.
    """
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180) or radius_m <= 0:
        raise ValueError("Invalid plant search coordinates or radius")
    cache_dir = Path(cache_dir)
    key = hashlib.sha256(f"{latitude:.7f},{longitude:.7f},{radius_m}".encode()).hexdigest()[:20]
    path = cache_dir / f"plant_name_{key}.json"
    if path.exists():
        response = json.loads(path.read_text(encoding="utf-8"))
    elif _state["failures"] >= MAX_CONSECUTIVE_FAILURES:
        if not _state["announced"]:
            warnings.warn(
                f"Overpass has failed {_state['failures']} times in a row; skipping the "
                f"remaining plant-name lookups. Re-run once it is reachable, or name "
                f"these plants by hand (Sec. V-B-2).", stacklevel=2)
            _state["announced"] = True
        return None, None
    else:
        query = (f'[out:json][timeout:25];nwr(around:{radius_m},{latitude},{longitude})'
                 '["power"~"^(plant|generator|substation)$"]["name"];out center tags;')
        try:
            result = session.post("https://overpass-api.de/api/interpreter", data={"data": query},
                                  headers={"User-Agent": "synthetic-grid-research/1.0"}, timeout=35)
            result.raise_for_status()
            response = result.json()
            if "elements" not in response or "remark" in response:
                raise ValueError("Overpass query incomplete")
        except (requests.RequestException, ValueError) as exc:
            _state["failures"] += 1
            warnings.warn(f"Plant-name query failed; manual lookup required: {exc}", stacklevel=2)
            return None, None
        _state["failures"] = 0
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response), encoding="utf-8")
    return nearest_named(latitude, longitude, _candidates(response), radius_m)
