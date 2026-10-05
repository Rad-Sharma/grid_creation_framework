"""Geographic candidate ranking with explicit uncertainty, reusable by region."""
import numpy as np


def match_station(stations, point, kv, interconnection=None, max_km=5., plant_geometry=None):
    candidates = stations.copy()
    candidates['distance_km'] = np.linalg.norm(candidates[['x','y']].to_numpy()-point, axis=1)/1000
    candidates = candidates[candidates.distance_km <= max_km]
    if np.isfinite(kv) and kv >= 69:
        candidates = candidates[candidates.base_kV.eq(kv)]
    if interconnection and 'interconnection' in candidates:
        candidates = candidates[candidates.interconnection.isin(['', interconnection])]
    if candidates.empty:
        return None, 'no voltage/interconnection-compatible candidate within mapping radius', []
    if plant_geometry is not None:
        from shapely.geometry import Point
        candidates['inside_plant'] = [plant_geometry.covers(Point(x,y))
                                      for x,y in candidates[['longitude','latitude']].to_numpy()]
    else:
        candidates['inside_plant'] = False
    candidates['score'] = candidates.distance_km + (~candidates.inside_plant)*max_km
    candidates = candidates.sort_values(['score','bus_i'])
    # Multiple voltage tiers of one station are also ambiguous if voltage is unknown.
    alternatives = candidates[['bus_i','distance_km','score','inside_plant']].head(5).to_dict('records')
    if len(candidates) > 1 and candidates.iloc[1].score-candidates.iloc[0].score < .25:
        return None, 'ambiguous candidates: score margin below 0.25 km-equivalent', alternatives
    return candidates.iloc[0], 'plant_footprint' if candidates.iloc[0].inside_plant else 'distance_voltage_screen', alternatives
