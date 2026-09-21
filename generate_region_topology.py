"""Shared regional/country OSM topology entry point, independent of Texas data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent/'create_new_model'))
import geopandas as gpd
import pandas as pd
from grid_topology import network
from model_quality import quality_report
from substation_audit import run_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--osm', type=Path, required=True, help='Overpass out body geom JSON')
    p.add_argument('--boundary', type=Path, required=True)
    p.add_argument('--metric-crs', required=True, help='Region-appropriate projected CRS for distance/length')
    p.add_argument('--evidence', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--min-kv', type=float, default=69.,
                   help='MIN_TRANSMISSION_VOLTAGE in kV (manuscript Table VI)')
    args = p.parse_args()
    boundary = gpd.read_file(args.boundary).to_crs(4326).geometry.union_all()
    evidence = json.loads(args.evidence.read_text()) if args.evidence else {}
    elements = json.loads(args.osm.read_text())['elements']
    buses, branches, audit = network(elements, boundary, evidence=evidence,
                                    metric_crs=args.metric_crs, min_kv=args.min_kv)
    args.output.mkdir(parents=True, exist_ok=True)
    buses.to_csv(args.output/'buses.csv', index=False)
    pd.DataFrame(branches).drop(columns='geometry', errors='ignore').to_csv(args.output/'branches.csv', index=False)
    audit.to_csv(args.output/'topology_uncertainty.csv', index=False)
    (args.output/'model_quality.json').write_text(json.dumps(quality_report(buses,branches,audit),indent=2))
    run_audit(elements,boundary,buses,pd.DataFrame(branches),args.output/'substation_review',
              args.metric_crs,min_kv=args.min_kv)


if __name__ == '__main__':
    main()
