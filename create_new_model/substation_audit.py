"""Repeatable physical-substation duplicate and coverage-gap review.

This module never merges electrical buses or invents missing equipment. Source
identity aliases are distinguished from spatial similarity and reviewed decisions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import transform
from shapely.strtree import STRtree

RULE_VERSION = '1.0'


def projected_transform(metric_crs):
    crs = CRS.from_user_input(metric_crs)
    if not crs.is_projected or any(not np.isclose(a.unit_conversion_factor,1.) for a in crs.axis_info):
        raise ValueError('A projected CRS with meter units is required')
    return Transformer.from_crs(4326,crs,always_xy=True).transform


def normalized(value):
    text = unicodedata.normalize('NFKD',str(value or '')).casefold()
    return ' '.join(re.findall(r'[a-z0-9]+',text))


def collect_substations(elements, boundary, metric_crs=5070, min_kv=None):
    from grid_topology import DEFAULT_MIN_KV, element_geometry, voltages
    min_kv = DEFAULT_MIN_KV if min_kv is None else min_kv
    forward = projected_transform(metric_crs)
    records, seen = [], set()
    for e in elements:
        tags = e.get('tags',{})
        if tags.get('power') != 'substation':
            continue
        source = f'{e["type"]}/{e["id"]}'
        if source in seen:
            raise ValueError(f'Duplicate OSM identity in input: {source}')
        seen.add(source)
        try:
            geom = element_geometry(e)
        except (KeyError,ValueError,TypeError):
            geom = None
        reason = ''
        if geom is None or geom.is_empty:
            reason = 'invalid_or_missing_geometry'
        elif not boundary.covers(geom.representative_point()):
            reason = 'outside_boundary'
        vs = sorted(set(voltages(tags.get('voltage'), min_kv)))
        if not reason and not vs:
            if tags.get('voltage'):
                reason = 'unknown_or_below_modeled_voltage'
            elif tags.get('substation') in ('distribution','minor_distribution'):
                reason = 'distribution_without_modeled_voltage'
        records.append(dict(osm_id=source,name=tags.get('name',''),ref=tags.get('ref',''),
            operator=tags.get('operator',''),voltage_kv=vs,substation_type=tags.get('substation',''),
            eligibility_reason=reason,eligible=not reason,
            geometry=transform(forward,geom) if geom is not None and not geom.is_empty else None,
            element=e))
    # A member is suppressed only if its parent can actually be represented.
    parents = defaultdict(list)
    for r in records:
        if r['eligible'] and r['element']['type']=='relation':
            for member in r['element'].get('members',[]):
                if member['type']=='way':
                    parents[f"way/{member['ref']}"].append(r['osm_id'])
    for r in records:
        r['parent_relations'] = sorted(parents[r['osm_id']])
    return records


def duplicate_candidates(records, radius_m=100., decisions=None):
    if not np.isfinite(radius_m) or radius_m<=0:
        raise ValueError('Duplicate radius must be positive')
    decisions = decisions or []
    sources = {r['osm_id'] for r in records}
    reviewed = {}
    for d in decisions:
        key = tuple(sorted((d['left'],d['right'])))
        if (key in reviewed or key[0]==key[1] or not set(key)<=sources or
            d.get('decision') not in ('same_site','separate_sites') or not str(d.get('source','')).strip()):
            raise ValueError('Review decisions require distinct existing IDs, a unique pair, decision and source')
        reviewed[key] = d
    usable = sorted([r for r in records if r['geometry'] is not None and
                     r['eligibility_reason'] != 'outside_boundary'],key=lambda r:r['osm_id'])
    tree = STRtree([r['geometry'] for r in usable])
    pairs = set(reviewed)
    lookup = {r['osm_id']:r for r in records}
    for i,a in enumerate(usable):
        for j in tree.query(a['geometry'].buffer(radius_m)):
            if j>i and a['geometry'].distance(usable[j]['geometry'])<=radius_m:
                pairs.add((a['osm_id'],usable[j]['osm_id']))
        for parent in a['parent_relations']:
            pairs.add(tuple(sorted((a['osm_id'],parent))))
    result = []
    for key in sorted(pairs):
        a,b = (lookup[k] for k in key)
        ga,gb = a['geometry'],b['geometry']
        if ga is None or gb is None:
            distance,overlap,contained = None,0.,False
        else:
            distance = float(ga.distance(gb))
            union_area = ga.union(gb).area
            overlap = float(ga.intersection(gb).area/union_area) if ga.area>0 and gb.area>0 and union_area else 0.
            contained = ((ga.geom_type=='Point' and gb.area>0 and gb.covers(ga)) or
                         (gb.geom_type=='Point' and ga.area>0 and ga.covers(gb)))
        same_name = bool(normalized(a['name'])) and normalized(a['name'])==normalized(b['name'])
        same_ref = bool(normalized(a['ref'])) and normalized(a['ref'])==normalized(b['ref'])
        same_operator = bool(normalized(a['operator'])) and normalized(a['operator'])==normalized(b['operator'])
        operator_conflict = bool(normalized(a['operator']) and normalized(b['operator'])) and not same_operator
        voltage_disjoint = bool(a['voltage_kv'] and b['voltage_kv']) and not set(a['voltage_kv'])&set(b['voltage_kv'])
        alias = a['osm_id'] in b['parent_relations'] or b['osm_id'] in a['parent_relations']
        # Proximity without corroboration is not sufficient to call a duplicate.
        if key not in reviewed and not (alias or overlap>0 or contained or same_name or same_ref):
            continue
        category = ('relation_member_alias' if alias else 'possible_distinct_voltage_yards' if voltage_disjoint
                    else 'conflicting_operators' if operator_conflict
                    else 'likely_duplicate' if (contained or overlap>=.8) and (same_name or same_ref)
                    else 'possible_duplicate_or_adjacent_sections')
        decision = reviewed.get(key,{})
        result.append(dict(left=key[0],right=key[1],left_name=a['name'],right_name=b['name'],
            left_operator=a['operator'],right_operator=b['operator'],
            left_voltage_kv=a['voltage_kv'],right_voltage_kv=b['voltage_kv'],
            distance_m=distance,polygon_iou=overlap,point_in_polygon=bool(contained),
            same_name=same_name,same_ref=same_ref,same_operator=same_operator,
            voltage_disjoint=bool(voltage_disjoint),category=category,
            decision=decision.get('decision','unreviewed'),decision_source=decision.get('source',''),
            topology_action='none; physical identity does not establish electrical connectivity'))
    return result


def missing_candidates(records,buses,branches,boundary,metric_crs=5070,cluster_m=250.,nearby_m=500.):
    if min(cluster_m,nearby_m)<=0 or not np.isfinite([cluster_m,nearby_m]).all():
        raise ValueError('Gap search distances must be positive')
    forward=projected_transform(metric_crs)
    if buses.empty:
        return []
    if buses.bus_i.duplicated().any() or not np.isfinite(buses[['longitude','latitude','base_kV']].to_numpy()).all():
        raise ValueError('Bus identities must be unique and coordinates/voltages finite')
    if not branches.empty and not np.isin(branches[['fbus','tbus']].to_numpy(),buses.bus_i).all():
        raise ValueError('Branch references unknown bus')
    border=transform(forward,boundary)
    neighbors=defaultdict(set)
    active_branches = branches
    if 'status' in branches:
        active_branches = branches[branches.status>0]
    for a,b in active_branches.reindex(columns=['fbus','tbus']).itertuples(index=False,name=None):
        neighbors[int(a)].add(int(b));neighbors[int(b)].add(int(a))
    candidates=[]
    for row in buses.sort_values('bus_i').itertuples():
        degree=len(neighbors[int(row.bus_i)])
        if row.kind!='osm_junction' or not (degree==1 or degree>=3):
            continue
        point=Point(forward(row.longitude,row.latitude))
        if not border.covers(point) or border.boundary.distance(point)<=nearby_m:
            continue
        candidates.append((int(row.bus_i),point,degree,float(row.base_kV)))
    if not candidates:
        return []
    xy=np.array([[p.x,p.y] for _,p,_,_ in candidates])
    edges=sorted(cKDTree(xy).query_pairs(cluster_m))
    f,t=zip(*edges) if edges else ([],[])
    _,labels=connected_components(csr_matrix((np.ones(len(edges)),(f,t)),shape=(len(xy),len(xy))),directed=False)
    stations=[r for r in records if r['geometry'] is not None and r['eligibility_reason']!='outside_boundary']
    tree=STRtree([r['geometry'] for r in stations])
    inverse=Transformer.from_crs(metric_crs,4326,always_xy=True).transform
    result=[]
    for label in np.unique(labels):
        indexes=np.flatnonzero(labels==label)
        members=[candidates[i] for i in indexes]
        center=Point(xy[indexes].mean(axis=0))
        nearby=set()
        for _,point,_,_ in members:
            for j in tree.query(point.buffer(nearby_m)):
                if stations[j]['geometry'].distance(point)<=nearby_m:
                    nearby.add(stations[j]['osm_id'])
        ids=sorted(v[0] for v in members)
        lon,lat=inverse(center.x,center.y)
        result.append(dict(candidate_id='buses:'+','.join(map(str,ids)),bus_ids=ids,
            longitude=lon,latitude=lat,terminal_count=len(ids),
            voltage_kv=sorted(set(v[3] for v in members)),degrees=[v[2] for v in members],
            cluster_radius_m=float(np.linalg.norm(xy[indexes]-xy[indexes].mean(axis=0),axis=1).max()),
            nearby_source_substations=sorted(nearby),
            category='existing_station_unmatched' if nearby else 'possible_missing_substation_or_line_connection',
            priority='review' if len(ids)>1 or max(v[2] for v in members)>=3 else 'low',
            interpretation='Candidate only: may be a line end, tower junction, mapping gap, or omitted station; no bus created'))
    return result


def reference_gaps(records,reference,metric_crs=5070,radius_m=500.,boundary=None):
    forward=projected_transform(metric_crs)
    if not np.isfinite(radius_m) or radius_m<=0:
        raise ValueError('Reference search radius must be positive')
    if reference.crs is None or 'reference_id' not in reference or 'source' not in reference:
        raise ValueError('Reference requires CRS, reference_id and source')
    if reference.reference_id.isna().any() or reference.reference_id.duplicated().any() or reference.source.fillna('').str.strip().eq('').any():
        raise ValueError('Reference IDs must be unique and sources nonempty')
    ref=reference.to_crs(metric_crs)
    border=transform(forward,boundary) if boundary is not None else None
    stations=[r for r in records if r['geometry'] is not None]
    tree=STRtree([r['geometry'] for r in stations])
    result=[]
    for r in ref.sort_values('reference_id').itertuples():
        if r.geometry is None or r.geometry.is_empty:
            raise ValueError('Reference contains missing geometry')
        if border is not None and not border.covers(r.geometry.representative_point()):
            result.append(dict(reference_id=str(r.reference_id),source=str(r.source),nearby_osm_ids=[],
                               status='outside_study_boundary',interpretation='Excluded from gap counts'))
            continue
        matches=sorted(stations[j]['osm_id'] for j in tree.query(r.geometry.buffer(radius_m))
                       if stations[j]['geometry'].distance(r.geometry)<=radius_m)
        result.append(dict(reference_id=str(r.reference_id),source=str(r.source),
            nearby_osm_ids=matches,status='spatial_candidate_match' if matches else 'not_found_within_search_radius',
            interpretation='Proximity is not verified identity; check reference date and study scope'))
    return result


def run_audit(elements,boundary,buses,branches,output,metric_crs=5070,decisions=None,reference=None,
              radius_m=100.,cluster_m=250.,nearby_m=500.,min_kv=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=collect_substations(elements,boundary,metric_crs,min_kv)
    duplicates=duplicate_candidates(records,radius_m,decisions)
    missing=missing_candidates(records,buses,branches,boundary,metric_crs,cluster_m,nearby_m)
    ref=reference_gaps(records,reference,metric_crs,nearby_m,boundary) if reference is not None else []
    inventory=[]
    for r in sorted(records,key=lambda r:r['osm_id']):
        inventory.append({k:v for k,v in r.items() if k not in ('geometry','element')})
    for name,rows in [('substation_inventory',inventory),('substation_duplicate_candidates',duplicates),
                      ('substation_gap_candidates',missing),('substation_reference_comparison',ref)]:
        (output/f'{name}.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
        if rows:
            pd.DataFrame([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()}
                          for row in rows]).to_csv(output/f'{name}.csv',index=False)
        else:
            # JSON always contains a valid empty list; remove only a stale CSV
            # from a prior run of this same report, not arbitrary user files.
            (output/f'{name}.csv').unlink(missing_ok=True)
    summary=dict(rule_version=RULE_VERSION,metric_crs=str(metric_crs),
        thresholds=dict(duplicate_radius_m=radius_m,endpoint_cluster_m=cluster_m,nearby_station_m=nearby_m),
        source_substation_records=len(records),eligible_station_records=sum(r['eligible'] for r in records),
        source_alias_records=sum(bool(r['parent_relations']) for r in records),
        duplicate_categories=dict(Counter(r['category'] for r in duplicates)),
        duplicate_pairs=len(duplicates),gap_candidates=len(missing),
        gap_categories=dict(Counter(r['category'] for r in missing)),
        independent_reference_supplied=reference is not None,
        reference_records=len(ref),reference_unmatched=sum(r['status']=='not_found_within_search_radius' for r in ref),
        automatic_spatial_merges=0,created_substations=0,
        limitations=['Nearby/overlapping records may be distinct yards or bus sections.',
                     'Endpoint clusters are hypotheses, not proof of missing substations.',
                     'Coverage cannot be established without an independent dated inventory.'])
    (output/'substation_audit_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('osm','boundary','buses','branches','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--metric-crs',required=True)
    p.add_argument('--decisions',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--duplicate-radius-m',type=float,default=100.)
    p.add_argument('--cluster-m',type=float,default=250.)
    p.add_argument('--nearby-m',type=float,default=500.)
    a=p.parse_args()
    boundary=gpd.read_file(a.boundary).to_crs(4326).geometry.union_all()
    summary=run_audit(json.loads(a.osm.read_text())['elements'],boundary,pd.read_csv(a.buses),
        pd.read_csv(a.branches),a.output,a.metric_crs,
        json.loads(a.decisions.read_text()) if a.decisions else None,
        gpd.read_file(a.reference) if a.reference else None,
        a.duplicate_radius_m,a.cluster_m,a.nearby_m)
    summary['input_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest()
        for name,path in [('osm',a.osm),('boundary',a.boundary),('buses',a.buses),('branches',a.branches),
                          ('decisions',a.decisions),('reference',a.reference)] if path}
    (a.output/'substation_audit_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
