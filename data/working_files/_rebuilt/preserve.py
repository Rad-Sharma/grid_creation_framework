"""Copy sandbox builds out of session-scoped temp into the repository.

The sandbox lives under Temp\claude\<session-id>\, which no later session can
find. These are rebuilds, so they go to data/working_files/_rebuilt/{name}/ and
never touch the checked-in cases in data/working_files/{region}/, nor any
earlier rebuild: pass an explicit destination name when a folder already exists.

Usage: python preserve.py --sandbox <path> region[=policy][:destination] ...
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEST = REPO / 'data' / 'working_files' / '_rebuilt'


def preserve(sandbox, region, policy, name, note):
    src = sandbox / region / 'data' / 'working_files' / region
    if not src.is_dir():
        return None
    out = DEST / name
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src.iterdir()):
        if path.is_file():
            shutil.copy2(path, out / path.name)
            copied.append(path.name)
    # The feasible cases, predictions and features live outside working_files.
    extras = [f'matpower_files/{region}_dc_feasible.m',
              f'matpower_files/{region}_ac_feasible.m',
              f'data/bus_data/processed/{region}_substation_predictions_v4.csv',
              f'data/bus_data/features/{region}_features_with_industrial.csv',
              'dispatch/dispatch_report.json']
    for rel in extras:
        s = sandbox / region / rel
        if s.is_file():
            shutil.copy2(s, out / s.name)
            copied.append(s.name)
    (out / 'REBUILD_INFO.json').write_text(json.dumps(dict(
        region=region,
        review_policy=policy,
        rebuilt_at=datetime.now(timezone.utc).isoformat(),
        note=note,
    ), indent=2), encoding='utf-8')
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sandbox', type=Path, required=True)
    parser.add_argument('--note', default='Rebuilt from OSM with the current pipeline. '
                        'NOT a reviewed model: Step 3 was answered automatically '
                        'under the stated policy.')
    parser.add_argument('targets', nargs='+',
                        help='region[=policy][:destination]')
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        spec, _, name = target.partition(':')
        region, _, policy = spec.partition('=')
        copied = preserve(args.sandbox, region, policy or 'assisted',
                          name or region, args.note)
        if copied is None:
            print(f'{region}: sandbox not found')
        else:
            cases = [n for n in copied if n.endswith('.m')]
            print(f'{region}: {len(copied)} file(s) -> _rebuilt/{name or region}/'
                  f'  [{len(cases)} .m: {", ".join(sorted(cases))}]')


if __name__ == '__main__':
    main()
