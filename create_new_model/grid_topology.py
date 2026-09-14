"""Shared OSM topology engine for regional and country builders.

Projection must be appropriate to the region. No transformer is created merely
because a station has multiple voltages unless legacy mode is explicit.
"""
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform, unary_union, polygonize
from shapely.strtree import STRtree
from interconnection_rules import mixed_interconnection_circuits, line_interconnection, normally_open

def voltages(value):
    result = []
    for part in str(value or '').split(';'):
        try:
            v = float(part)
            if 69000 <= v <= 765000:
                result.append(v / 1000)
        except ValueError:
            pass
    return result


def element_geometry(e):
    if e['type'] == 'node':
        return Point(e['lon'], e['lat'])
    if e['type'] == 'way':
        pts = [(p['lon'], p['lat']) for p in e.get('geometry', [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            return Polygon(pts).buffer(0)
        return LineString(pts) if len(pts) >= 2 else None
    outer, inner = [], []
    for member in e.get('members', []):
        pts = [(p['lon'], p['lat']) for p in member.get('geometry', [])]
        if len(pts) >= 2:
            (inner if member.get('role') == 'inner' else outer).append(LineString(pts))
    geometry = unary_union(list(polygonize(outer)))
    if inner:
        geometry = geometry.difference(unary_union(list(polygonize(inner))))
    return geometry if not geometry.is_empty else None


def line_parameters(kv, km, circuits=1):
    """Manuscript Eq. 3 on 100 MVA; parallel circuits form one equivalent."""
    return (1115 * kv**-2.9870 * km / circuits,
            52.86 * kv**-2.0342 * km / circuits,
            .000049 * kv**.6896 * km * circuits,
            .018429 * kv**1.8998 * circuits)


def asynchronous_station(tags):
    """Do not collapse the AC terminals of an explicitly tagged converter."""
    frequency = {v.strip() for v in str(tags.get('frequency', '')).split(';')}
    return (tags.get('substation') == 'converter' or bool(tags.get('converter'))
            or '0' in frequency)


def network(elements, boundary, snap_m=150., *, evidence=None, transformer_policy="reviewed",
            switchable_stations=None, metric_crs=5070):
    """Only shared OSM vertices and nearby compatible station footprints connect.

    Geometric line crossings are not connections. Unknown-voltage lines are
    omitted and audited; unknown station tiers can be inferred from attached lines.
    """
    crs = CRS.from_user_input(metric_crs)
    if not crs.is_projected or any(not np.isclose(axis.unit_conversion_factor,1.) for axis in crs.axis_info):
        raise ValueError('Topology requires a projected CRS with meter units')
    FORWARD = Transformer.from_crs(4326, metric_crs, always_xy=True).transform
    INVERSE = Transformer.from_crs(metric_crs, 4326, always_xy=True).transform
    evidence = evidence or {}
    switchable_stations = switchable_stations or {}
    if transformer_policy not in ("reviewed", "legacy"):
        raise ValueError("Unknown transformer policy")
    border = transform(FORWARD, boundary)
    stations, omitted, segments = [], [], []
    station_relations = [e for e in elements if e['type'] == 'relation' and e.get('tags', {}).get('power') == 'substation']
    member_ids = {m['ref'] for e in station_relations for m in e.get('members', []) if m['type'] == 'way'}
    for e in elements:
        tags = e.get('tags', {})
        if tags.get('power') != 'substation' or (e['type'] == 'way' and e['id'] in member_ids):
            continue
        geom = element_geometry(e)
        if geom is None or not boundary.covers(geom.representative_point()):
            continue
        vs = sorted(set(voltages(tags.get('voltage'))))
        if not vs and tags.get('voltage'):
            continue
        if not vs and tags.get('substation') in ('distribution', 'minor_distribution'):
            continue
        stations.append(dict(osm_id=f'{e["type"]}/{e["id"]}',
                             name=tags.get('name') or tags.get('ref') or f'OSM {e["id"]}',
                             kv=vs, asynchronous=asynchronous_station(tags),
                             switchable=f'{e["type"]}/{e["id"]}' in switchable_stations,
                             geometry=transform(FORWARD, geom)))
    tree = STRtree([s['geometry'] for s in stations])
    print('Candidate substations:', len(stations), flush=True)
    buses, bus_keys, attached = [], {}, defaultdict(set)
    station_systems, open_sites = defaultdict(set), set()

    def bus_at(xy, kv, identity, system, way_id):
        point = Point(xy)
        candidates = tree.query(point.buffer(snap_m))
        eligible = [int(i) for i in candidates if (not stations[i]['kv'] or kv in stations[i]['kv'])
                    and stations[i]['geometry'].distance(point) <= snap_m]
        if eligible:
            # A converter footprint must not be bypassed by snapping to an
            # adjacent conventional switchyard within the tolerance.
            converters = [i for i in eligible if stations[i]['asynchronous'] or stations[i]['switchable']]
            site = min(converters or eligible, key=lambda i: (stations[i]['geometry'].distance(point), i))
            converter = stations[site]['asynchronous'] or stations[site]['switchable']
            # Keep terminal identity until every incident circuit and open tie
            # is known. Early station collapse cannot be undone reliably.
            key = ('terminal', site, kv, identity, system or way_id)
            if system:
                station_systems[site].add(system)
            center = point if converter else stations[site]['geometry'].representative_point()
            coord = (center.x, center.y)
            kind = 'converter_terminal_open' if converter else 'substation'
            if stations[site]['switchable']:
                kind = 'switchable_terminal_open'
            name = stations[site]['name']
            source = stations[site]['osm_id']
            inferred = kv not in stations[site]['kv']
        else:
            # A shared tower/node is not an electrical tie between explicitly
            # different interconnections. Unknown circuits at a conflicting
            # node remain separate pending conductor-level mapping.
            if identity[0] == 'osm_node' and len(node_systems[identity[1]]) > 1:
                identity = (identity, system or ('unresolved_way', way_id))
            key = ('vertex', identity, kv)
            coord, site, inferred = xy, None, False
            kind = 'boundary_open' if border.boundary.distance(point) < 1 else 'osm_junction'
            name, source = kind, ''
        if key not in bus_keys:
            idx = len(buses)
            bus_keys[key] = idx
            lon, lat = INVERSE(*coord)
            buses.append(dict(bus_i=idx + 1, name=name, base_kV=kv, longitude=lon,
                              latitude=lat, kind=kind, osm_id=source, station=site,
                              inferred_voltage=inferred, x=coord[0], y=coord[1]))
            buses[-1].update(terminal_identity=identity, terminal_system=system or '',
                             section='', connection_evidence='')
            if site is not None and kind == 'substation':
                attached[site].add(idx)
        return bus_keys[key]

    # Split at explicit shared OSM nodes, including interior nodes of ways.
    node_uses = Counter()
    node_systems = defaultdict(set)
    lines = []
    for e in elements:
        tags = e.get('tags', {})
        if e['type'] != 'way' or tags.get('power') not in ('line', 'cable'):
            continue
        levels = voltages(tags.get('voltage'))
        reason = None
        if normally_open(tags):
            reason = 'explicitly normally-open connection; excluded from base topology'
        elif '0' in str(tags.get('frequency', '')).split(';') or str(tags.get('current', '')).upper() == 'DC':
            reason = 'DC line requires explicit converter/interconnection model'
        elif mixed_interconnection_circuits(tags):
            reason = 'mixed ERCOT/Eastern circuits: withheld until circuit-specific connectivity is available'
        elif not levels:
            reason = 'unknown or below-69-kV voltage'
        if reason:
            if normally_open(tags):
                for p in (e.get('geometry', [])[:1] + e.get('geometry', [])[-1:]):
                    point = Point(FORWARD(p['lon'], p['lat']))
                    for site in tree.query(point.buffer(snap_m)):
                        if stations[site]['geometry'].distance(point) <= snap_m:
                            open_sites.add(int(site))
            omitted.append(dict(osm_id=f'way/{e["id"]}', reason=reason, voltage=tags.get('voltage', '')))
            continue
        lines.append(e)
        node_uses.update(set(e.get('nodes', [])))
        system = line_interconnection(tags)
        if system:
            for node in e.get('nodes', []):
                node_systems[node].add(system)
    for number, e in enumerate(lines):
        pts = [(p['lon'], p['lat']) for p in e['geometry']]
        if len(pts) < 2:
            continue
        cuts = [0] + [i for i, node in enumerate(e['nodes'][1:-1], 1) if node_uses[node] > 1] + [len(pts)-1]
        counts = Counter(voltages(e['tags'].get('voltage')))
        if len(counts) == 1:
            # Explicit circuits or cable count can describe repeated same-voltage circuits.
            try:
                total = int(e['tags'].get('circuits') or int(e['tags'].get('cables', 3)) // 3)
                kv = next(iter(counts))
                counts[kv] = max(counts[kv], min(max(total, 1), 8))
            except (ValueError, TypeError):
                pass
        for start, end in zip(cuts[:-1], cuts[1:]):
            geo = transform(FORWARD, LineString(pts[start:end+1]))
            clipped = geo if border.covers(geo) else geo.intersection(border)
            parts = list(clipped.geoms) if hasattr(clipped, 'geoms') else [clipped]
            for part in parts:
                if part.geom_type != 'LineString' or part.length < 1:
                    continue
                for kv, circuits in counts.items():
                    def identity(xy):
                        for endpoint in (start, end):
                            if Point(xy).distance(Point(FORWARD(*pts[endpoint]))) < 1e-5:
                                return ('osm_node', e['nodes'][endpoint])
                        # Clipping creates an open terminal, not a shared node.
                        return ('clipped', e['id'], start, end, tuple(xy))
                    system = line_interconnection(e['tags'])
                    f = bus_at(part.coords[0], kv, identity(part.coords[0]), system, e['id'])
                    t = bus_at(part.coords[-1], kv, identity(part.coords[-1]), system, e['id'])
                    if f == t:
                        continue
                    r, x, b, rate = line_parameters(kv, part.length / 1000, circuits)
                    segments.append(dict(fbus=f+1, tbus=t+1, r=r, x=x, b=b, rateA=rate,
                                         kind='line', osm_id=f'way/{e["id"]}', length_km=part.length/1000,
                                         voltage_kV=kv, circuits=circuits,
                                         transmission_system=e['tags'].get('transmission_system', ''),
                                         geometry=transform(INVERSE, part).simplify(.001)))
        if number % 5000 == 0:
            print('Processed OSM ways:', number, '/', len(lines), flush=True)
    # Collapse only compatible terminal sections after examining all circuits.
    remap, merged, keys = {}, [], {}
    for old, bus in enumerate(buses):
        site = bus['station']
        if site is None:
            key = ('vertex', old)
        else:
            station = stations[site]
            spec = evidence.get('stations', {}).get(station['osm_id'], {})
            identity = bus['terminal_identity']
            node = str(identity[1]) if identity[0] == 'osm_node' else ''
            explicit = spec.get('node_sections', {}).get(node)
            if explicit is not None:
                if not spec.get('source'):
                    raise ValueError('Station section evidence needs a source')
                section = str(explicit)
                bus['connection_evidence'] = spec['source']
            elif (site in open_sites or station['asynchronous'] or station['switchable']):
                section = repr((identity, bus['terminal_system']))
                bus['connection_evidence'] = 'unresolved terminal; no station-wide connection inferred'
            elif len(station_systems[site]) > 1:
                section = bus['terminal_system'] or repr(identity)
                bus['connection_evidence'] = 'separated by conflicting interconnection tags'
            else:
                section = 'default'
                bus['connection_evidence'] = 'synthetic common bus; no contradictory tags'
            bus['section'] = section
            bus['interconnection'] = (bus['terminal_system'] or
                (next(iter(station_systems[site])) if len(station_systems[site]) == 1 else ''))
            key = ('station', site, bus['base_kV'], section)
        if key not in keys:
            keys[key] = len(merged) + 1
            bus['bus_i'] = keys[key]
            merged.append(bus)
        remap[old+1] = keys[key]
    buses = merged
    for segment in segments:
        segment['fbus'], segment['tbus'] = remap[segment['fbus']], remap[segment['tbus']]
    segments = [s for s in segments if s['fbus'] != s['tbus']]
    attached = defaultdict(set)
    for i, bus in enumerate(buses):
        if bus['station'] is not None and bus['kind'] == 'substation':
            attached[bus['station']].add(i)
    # Unverified transformer candidates are reported, never silently installed.
    for site, ids in attached.items():
        ordered = sorted(ids, key=lambda i: buses[i]['base_kV'])
        for lo, hi in zip(ordered[:-1], ordered[1:]):
            if buses[lo]['base_kV'] == buses[hi]['base_kV']:
                continue
            if (transformer_policy != 'legacy' or buses[lo]['section'] != 'default'
                    or buses[hi]['section'] != 'default'):
                omitted.append(dict(osm_id=stations[site]['osm_id'],
                    reason='unverified transformer candidate; requires equipment evidence',
                    voltage=f"{buses[lo]['base_kV']};{buses[hi]['base_kV']}"))
                continue
            segments.append(dict(fbus=lo+1, tbus=hi+1, r=.00002, x=.0016, b=0., rateA=1000.,
                                 kind='assumed_transformer', osm_id=stations[site]['osm_id'],
                                 length_km=0., voltage_kV=buses[hi]['base_kV'], circuits=1,
                                 transmission_system='',
                                 geometry=None))
    for equipment in evidence.get('connections', []):
        if not equipment.get('source') or equipment.get('kind') not in ('transformer', 'switch'):
            raise ValueError('Reviewed connection needs kind and source')
        if not isinstance(equipment.get('closed',True),bool):
            raise ValueError('Connection closed state must be a JSON boolean')
        ends = []
        for side in ('from', 'to'):
            spec = equipment[side]
            found = [b['bus_i'] for b in buses if b['osm_id'] == spec['station']
                     and b['base_kV'] == spec['kv'] and b['section'] == spec['section']]
            if len(found) != 1:
                raise ValueError(f'Reviewed connection endpoint must resolve uniquely: {spec}')
            ends.append(found[0])
        r, x, rate = (float(equipment[k]) for k in ('r_pu', 'x_pu', 'rate_mva'))
        if not np.isfinite([r,x,rate]).all() or r < 0 or x <= 0 or rate <= 0:
            raise ValueError('Equipment requires finite r>=0, x>0 and positive rating on 100 MVA base')
        if not equipment.get('closed', True):
            continue
        segments.append(dict(fbus=ends[0], tbus=ends[1], r=r, x=x, b=0., rateA=rate,
            kind='reviewed_'+equipment['kind'], osm_id=equipment.get('id', ''), length_km=0.,
            voltage_kV=equipment['to']['kv'], circuits=1, transmission_system='',
            evidence_source=equipment['source'], geometry=None))
    unused = [dict(osm_id=s['osm_id'], reason=('switchable station: terminals open; switching state unavailable'
                                            if s['switchable'] else 'asynchronous converter: terminals retained open; transfer unmodeled'
                                            if s['asynchronous'] else 'no matched modeled line'), voltage=str(s['kv']))
              for i, s in enumerate(stations) if i not in attached]
    return pd.DataFrame(buses), segments, pd.DataFrame(omitted + unused)
