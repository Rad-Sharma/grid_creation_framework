"""Shared MATPOWER serialization and island extraction."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from power_network import validate_case

def islands(case):
    ids = {int(v): i for i, v in enumerate(case['bus'][:, 0])}
    br = case['branch'][case['branch'][:, 10] > 0]
    f = np.array([ids[int(v)] for v in br[:, 0]], dtype=int)
    t = np.array([ids[int(v)] for v in br[:, 1]], dtype=int)
    return connected_components(csr_matrix((np.ones(len(f)), (f, t)),
                                shape=(len(ids), len(ids))), directed=False)[1]


def island_case(case, labels, label):
    members = np.flatnonzero(labels == label)
    ids = case['bus'][members, 0].astype(int)
    mapping = {v: i for i, v in enumerate(ids)}
    gm = np.isin(case['gen'][:, 0], ids) & (case['gen'][:, 7] > 0)
    bm = (np.isin(case['branch'][:, 0], ids) & np.isin(case['branch'][:, 1], ids)
          & (case['branch'][:, 10] > 0))
    sub = dict(version='2', baseMVA=case['baseMVA'], bus=case['bus'][members].copy(),
               gen=case['gen'][gm].copy(), branch=case['branch'][bm].copy())
    sub['bus'][:, 0] = np.arange(len(ids))
    sub['gen'][:, 0] = [mapping[int(v)] for v in sub['gen'][:, 0]]
    for col in (0, 1):
        sub['branch'][:, col] = [mapping[int(v)] for v in sub['branch'][:, col]]
    return sub, ids, np.flatnonzero(bm), np.flatnonzero(gm)


def write_case(case,path,label):
    validate_case(case)
    with path.open('w',encoding='utf-8') as out:
        out.write(f'function mpc = {path.stem}\n% {label}\n% Synthetic case; consult accompanying provenance and validation reports.\n')
        out.write(f"mpc.version = '2';\nmpc.baseMVA = {float(case['baseMVA']):.15g};\n")
        for key in ('bus','gen','branch','gencost'):
            out.write(f'mpc.{key} = [\n')
            np.savetxt(out,case[key],fmt='%.15g',delimiter='\t',newline=';\n')
            out.write('];\n')
        out.write('end\n')
