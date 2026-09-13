"""Section V.B.1: cached nearby OSM-name inference for unnamed plants."""
import hashlib
import json
from pathlib import Path
import warnings

from geopy.distance import geodesic
import requests


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
            warnings.warn(f"Plant-name query failed; manual lookup required: {exc}", stacklevel=2)
            return None, None
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response), encoding="utf-8")
    candidates = []
    for element in response.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name or tags.get("power") not in {"plant", "generator", "substation"}:
            continue
        center = element.get("center", element)
        if "lat" not in center or "lon" not in center:
            continue
        distance = geodesic((latitude, longitude), (center["lat"], center["lon"])).meters
        if distance <= radius_m:
            osm_id = f"{element.get('type', '?')}/{element.get('id', '?')}"
            candidates.append((distance, osm_id, name))
    if not candidates:
        return None, None
    _, osm_id, name = min(candidates)
    return f"[{name}]", osm_id
