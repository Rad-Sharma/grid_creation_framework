"""Create an explicit hourly operating case, retaining input provenance.

No measured data is supplied by this module. JSON inputs must provide sources,
sector loads, generator availability, and any storage/interchange schedules.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SECTORS = {'residential','commercial','industrial','other'}


def allocate_sectors(buses, counties, totals, weights):
    """Allocate supplied county/sector totals using supplied bus-specific weights."""
    import geopandas as gpd
    if totals.duplicated(['GEOID','sector']).any() or weights.duplicated(['bus_i','sector']).any():
        raise ValueError('Duplicate sector total or bus weight')
    if not set(totals.sector).issubset(SECTORS) or not set(weights.sector).issubset(SECTORS):
        raise ValueError('Unknown load sector')
    if not totals.GEOID.astype(str).isin(counties.GEOID.astype(str)).all():
        raise ValueError('Unknown county in sector totals')
    if not weights.bus_i.isin(buses.bus_i).all():
        raise ValueError('Unknown bus in sector weights')
    for frame,col in ((totals,'mw'),(weights,'weight')):
        if not np.isfinite(frame[col]).all() or (frame[col]<0).any():
            raise ValueError('Sector totals and weights must be finite and nonnegative')
        if 'source' not in frame or frame.source.fillna('').str.strip().eq('').any():
            raise ValueError('Sector inputs require sources')
    points = gpd.GeoDataFrame(buses[['bus_i']],geometry=gpd.points_from_xy(buses.longitude,buses.latitude),crs=4326)
    joined = gpd.sjoin(points,counties[['GEOID','geometry']],how='left',predicate='intersects')
    joined = joined.sort_values('GEOID').drop_duplicates('bus_i')
    allocation = weights.merge(joined[['bus_i','GEOID']],on='bus_i',validate='many_to_one')
    output = []
    for _,r in totals.iterrows():
        local = allocation[allocation.GEOID.astype(str).eq(str(r.GEOID)) & allocation.sector.eq(r.sector)]
        mass = local.weight.sum()
        if r.mw > 0 and mass <= 0:
            raise ValueError(f'Missing positive allocation weights: {r.GEOID}/{r.sector}')
        for _,w in local.iterrows():
            output.append(dict(bus_i=int(w.bus_i),sector=r.sector,mw=float(r.mw*w.weight/mass) if mass else 0.,
                               GEOID=str(r.GEOID),source=r.source,weight_source=w.source))
    detail = pd.DataFrame(output,columns=['bus_i','sector','mw','GEOID','source','weight_source'])
    load = detail.groupby('bus_i').mw.sum().reindex(buses.bus_i,fill_value=0).to_numpy()
    if not np.isclose(load.sum(),totals.mw.sum(),atol=1e-6,rtol=0):
        raise ValueError('Sector allocation failed to preserve total')
    return load,detail


def apply_scenario(case, spec):
    if not str(spec.get('source','')).strip() or not spec.get('timestamp'):
        raise ValueError('Scenario needs source and timestamp')
    timestamp = pd.Timestamp(spec['timestamp'])
    if timestamp.tzinfo is None:
        raise ValueError('Scenario timestamp requires timezone')
    duration = float(spec['duration_hours'])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Scenario duration must be positive')
    result = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in case.items()}
    b,g = result['bus'],result['gen']
    bus_index = {int(v):i for i,v in enumerate(b[:,0])}
    if spec.get('replace_all_loads') is not True:
        raise ValueError('Explicit replace_all_loads=true is required; unspecified bus loads become zero')
    b[:,2:4] = 0.
    seen = set()
    for r in spec['sector_loads']:
        key = (int(r['bus']),r['sector'])
        if key in seen or r['sector'] not in SECTORS or key[0] not in bus_index:
            raise ValueError('Duplicate or unknown bus/sector')
        seen.add(key)
        mw,pf = float(r['mw']),float(r['power_factor'])
        if not np.isfinite([mw,pf]).all() or mw<0 or not 0<pf<=1:
            raise ValueError('Invalid sector demand/power factor')
        b[bus_index[key[0]],2] += mw
        b[bus_index[key[0]],3] += mw*np.tan(np.arccos(pf))
    seen = set()
    for r in spec['generator_availability']:
        row = int(r['row_1based'])-1
        factor = float(r['fraction'])
        if row in seen or not 0<=row<len(g) or not np.isfinite(factor) or not 0<=factor<=1:
            raise ValueError('Invalid or duplicate generator availability')
        seen.add(row)
        g[row,8] *= factor
        if g[row,9]>g[row,8]:
            raise ValueError('Availability conflicts with minimum generation')
        g[row,1] = np.clip(g[row,1],g[row,9],g[row,8])
    if not set(np.flatnonzero(g[:,7]>0)).issubset(seen):
        raise ValueError('Availability must cover every active generator')
    storage_report = []
    def injection(row):
        bus = int(row['bus'])
        if bus not in bus_index or not row.get('source'):
            raise ValueError('Scheduled injection requires known bus and source')
        mw,qmin,qmax = (float(row[k]) for k in ('mw','qmin_mvar','qmax_mvar'))
        if not np.isfinite([mw,qmin,qmax]).all() or qmin>qmax:
            raise ValueError('Invalid scheduled injection bounds')
        gen = np.zeros((1,result['gen'].shape[1]))
        gen[0,[0,1,3,4,5,6,7,8,9]] = [bus,mw,qmax,qmin,1,result['baseMVA'],1,mw,mw]
        result['gen'] = np.vstack([result['gen'],gen])
        cost = np.zeros((1,result['gencost'].shape[1]))
        cost[0,[0,3]] = [2,3]
        result['gencost'] = np.vstack([result['gencost'],cost])
    for r in spec.get('storage',[]):
        p,e,cap,charge,discharge,eta_c,eta_d = (float(r[k]) for k in
            ('mw','initial_mwh','capacity_mwh','charge_limit_mw','discharge_limit_mw','charge_efficiency','discharge_efficiency'))
        if (not np.isfinite([p,e,cap,charge,discharge,eta_c,eta_d]).all() or cap<=0 or
            not 0<=e<=cap or min(charge,discharge)<0 or not 0<eta_c<=1 or not 0<eta_d<=1 or
            not -charge<=p<=discharge):
            raise ValueError('Invalid storage state, power or efficiency')
        final = e-max(p,0)*duration/eta_d+max(-p,0)*duration*eta_c
        if not -1e-8<=final<=cap+1e-8:
            raise ValueError('Storage schedule violates energy bounds')
        injection(r)
        storage_report.append(dict(bus=r['bus'],initial_mwh=e,final_mwh=final))
    for r in spec.get('interchange',[]):
        injection(r)  # positive MW is import; negative is export
    return result,dict(timestamp=timestamp.isoformat(),duration_hours=duration,source=spec['source'],
        demand_mw=float(b[:,2].sum()),storage=storage_report,
        limitation='Fixed interchange injections are not a validated external-network equivalent; this is one interval, not chronology')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    from ensure_feasibility import load_matpower
    from case_utils import write_case
    spec = json.loads(args.scenario.read_text())
    digest = hashlib.sha256(args.case.read_bytes()).hexdigest()
    if spec.get('source_case_sha256') != digest:
        raise ValueError('Scenario case hash mismatch; bus/generator row IDs may have changed')
    result,report = apply_scenario(load_matpower(args.case),spec)
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError('Use an empty output directory')
    write_case(result,args.output/'scenario.m','Explicit operating scenario; requires constrained dispatch validation')
    report.update(case_sha256=digest,scenario_sha256=hashlib.sha256(args.scenario.read_bytes()).hexdigest())
    (args.output/'scenario_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
