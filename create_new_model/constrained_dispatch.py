"""AC optimal dispatch with fixed loads and equipment limits, per island.

Exports only independently checked solutions. Solver failure or timeout is not
proof of infeasibility. No shunts, ratings, or generator bounds are relaxed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
from pypower.api import ppoption, runopf
from pypower.makeYbus import makeYbus


def assess(case):
    b, g, br = (case[k].copy() for k in ('bus','gen','branch'))
    if not all(np.isfinite(a).all() for a in (b,g,br)):
        return dict(passes=False, error='nonfinite solution')
    lookup = {int(v):i for i,v in enumerate(b[:,0])}
    b[:,0] = np.arange(len(b))
    g[:,0] = [lookup[int(v)] for v in g[:,0]]
    for col in (0,1):
        br[:,col] = [lookup[int(v)] for v in br[:,col]]
    v = b[:,7]*np.exp(1j*np.deg2rad(b[:,8]))
    y,yf,yt = makeYbus(case['baseMVA'], b, br)
    inj = -b[:,2]-1j*b[:,3]
    active = g[:,7] > 0
    np.add.at(inj, g[active,0].astype(int), g[active,1]+1j*g[active,2])
    residual = v*np.conj(y@v)*case['baseMVA']-inj
    f,t = br[:,:2].astype(int).T
    sf = case['baseMVA']*v[f]*np.conj(yf@v)
    st = case['baseMVA']*v[t]*np.conj(yt@v)
    limited = (br[:,10] > 0) & (br[:,5] > 0)
    angle = b[f,8]-b[t,8]
    low = (br[:,10]>0)&(br[:,11]>-360)&((br[:,11]!=0)|(br[:,12]!=0))
    high = (br[:,10]>0)&(br[:,12]<360)&((br[:,11]!=0)|(br[:,12]!=0))
    def maximum(values):
        return float(np.max(np.maximum(values,0),initial=0))
    metrics = dict(
        residual_mva=maximum(abs(residual)),
        thermal_mva=maximum(np.maximum(abs(sf[limited]),abs(st[limited]))-br[limited,5]),
        voltage_pu=maximum(np.maximum(b[:,7]-b[:,11],b[:,12]-b[:,7])),
        pg_mw=maximum(np.maximum(g[active,1]-g[active,8],g[active,9]-g[active,1])),
        qg_mvar=maximum(np.maximum(g[active,2]-g[active,3],g[active,4]-g[active,2])),
        offline_generation_mva=maximum(abs(g[~active,1:3])),
        angle_deg=max(maximum(br[low,11]-angle[low]),maximum(angle[high]-br[high,12])))
    metrics['passes'] = all(v <= (1e-6 if k=='voltage_pu' else 1e-4) for k,v in metrics.items())
    return metrics


def solve_island(case):
    candidate = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in case.items()}
    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter('always')
        solved = runopf(candidate, ppoption(VERBOSE=0,OUT_ALL=0,OPF_ALG=560,
                        PDIPM_MAX_IT=100,OPF_VIOLATION=1e-8,PDIPM_FEASTOL=1e-8))
    report = dict(solver_success=bool(solved['success']),
                  warnings=list(dict.fromkeys(str(w.message) for w in notes))[:5])
    if not solved['success']:
        return None, dict(report, passes=False)
    report.update(assess(solved))
    # Independently ensure an optimizer never changed the physical problem.
    for key, cols in [('bus',[2,3,4,5,11,12]),('gen',[3,4,7,8,9]),('branch',list(range(13)))]:
        if not np.array_equal(solved[key][:,cols],case[key][:,cols]):
            return None, dict(report, passes=False,error='solver changed fixed inputs')
    return (solved if report['passes'] else None), report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--timeout',type=float,default=120.,help='Wall seconds per island')
    p.add_argument('--total-timeout',type=float,default=600.,help='Overall solver wall-time budget; untested islands are reported')
    p.add_argument('--worker',nargs=3,metavar=('INPUT','OUTPUT','REPORT'))
    args = p.parse_args()
    if args.worker:
        inp,out,report = map(Path,args.worker)
        try:
            case = dict(np.load(inp,allow_pickle=False))
            case['baseMVA'] = float(case['baseMVA'])
            solved,result = solve_island(case)
            if solved is not None:
                np.savez_compressed(out,**{k:solved[k] for k in ('baseMVA','bus','gen','branch','gencost')})
        except Exception as exc:
            result = dict(passes=False,error=f'{type(exc).__name__}: {exc}')
        report.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return
    if not args.case or not args.output or not np.isfinite([args.timeout,args.total_timeout]).all() or min(args.timeout,args.total_timeout) <= 0:
        p.error('--case, --output and positive --timeout are required')
    from case_utils import islands, island_case
    from ensure_feasibility import load_matpower
    from case_utils import write_case
    case = load_matpower(args.case)
    labels = islands(case)
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        p.error('Use an empty output directory so failed runs cannot leave stale solved cases')
    results = []
    deadline = time.monotonic()+args.total_timeout
    combined = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in case.items()}
    with tempfile.TemporaryDirectory(prefix='grid_dispatch_') as temporary:
        temp = Path(temporary)
        for label in np.unique(labels):
            sub, ids, bi, gi = island_case(case,labels,label)
            sub['gencost'] = case['gencost'][gi].copy()
            report = dict(component=int(label),buses=len(ids),demand_mw=float(sub['bus'][:,2].sum()))
            if not len(gi):
                report.update(passes=False,status='no_generation')
            elif sub['gen'][:,8].sum() < sub['bus'][:,2].sum()-1e-4:
                report.update(passes=False,status='active_capacity_shortfall')
            elif time.monotonic() >= deadline:
                report.update(passes=False,status='not_attempted_budget_exhausted')
            else:
                inp,out,rep = (temp/name for name in ('input.npz','output.npz','report.json'))
                for path in (out,rep):
                    path.unlink(missing_ok=True)
                np.savez_compressed(inp,**sub)
                try:
                    proc = subprocess.run([sys.executable,__file__,'--worker',str(inp),str(out),str(rep)],
                        capture_output=True,text=True,timeout=min(args.timeout,max(.01,deadline-time.monotonic())))
                    if proc.returncode or not rep.exists():
                        report.update(passes=False,status='worker_error',error=proc.stderr[-1000:])
                    else:
                        report.update(json.loads(rep.read_text()))
                        report['status'] = 'validated' if report['passes'] else 'solver_or_validation_failed'
                    if report['passes']:
                        solved = dict(np.load(out,allow_pickle=False))
                        # Restore external IDs for export and full-case assembly.
                        solved['bus'][:,0] = ids
                        solved['gen'][:,0] = ids[solved['gen'][:,0].astype(int)]
                        solved['branch'][:,:2] = ids[solved['branch'][:,:2].astype(int)]
                        write_case(solved,args.output/f'island_{label}_ac.m','Fixed-equipment constrained AC dispatch')
                        rows = np.flatnonzero(labels==label)
                        combined['bus'][rows,7:9] = solved['bus'][:,7:9]
                        combined['gen'][gi,1:3] = solved['gen'][:,1:3]
                except subprocess.TimeoutExpired:
                    report.update(passes=False,status='timeout')
            results.append(report)
            if label % 20 == 0 or report.get('status')=='timeout':
                print(f'Island {label}: {report["status"]}',flush=True)
        full = all(r['passes'] for r in results)
        if full:
            check = assess(combined)
            full = check['passes']
            if full:
                write_case(combined,args.output/'full_ac.m','Full-case fixed-equipment AC dispatch')
        summary = dict(case_sha256=hashlib.sha256(args.case.read_bytes()).hexdigest(),
            full_case_validated=full, equipment_relaxed=False, load_reduced=False,
            limitation='Timeout or solver failure is not an infeasibility proof; feasibility is not geographic accuracy',
            islands=results)
        (args.output/'dispatch_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(f'Validated {sum(r["passes"] for r in results)}/{len(results)} islands; full case: {full}',flush=True)


if __name__ == '__main__':
    main()
