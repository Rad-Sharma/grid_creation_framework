"""Copy sandbox builds out of session-scoped temp into the repository.

The sandbox lives under Temp\\claude\\<session-id>\\, which no later session can
find. These are rebuilds, so they go to data/working_files/_rebuilt/{region}/
and never touch the checked-in cases in data/working_files/{region}/.
"""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(r'C:\Users\rad_s\OneDrive\Desktop\Synthetic Grids\grid_creation_framework')
SB = Path(r'C:\Users\rad_s\AppData\Local\Temp\claude'
          r'\C--Users-rad-s-OneDrive-Desktop-Synthetic-Grids-grid-creation-framework'
          r'\3875e432-6f79-4699-b299-cb672353ec9d\scratchpad\sandbox')
DEST = REPO / 'data' / 'working_files' / '_rebuilt'


def preserve(region, policy):
    src = SB / region / 'data' / 'working_files' / region
    if not src.is_dir():
        return None
    out = DEST / region
    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(src.iterdir()):
        if path.is_file():
            shutil.copy2(path, out / path.name)
            copied.append(path.name)
    # Predictions and features live outside working_files.
    for rel in (f'data/bus_data/processed/{region}_substation_predictions_v4.csv',
                f'data/bus_data/features/{region}_features_with_industrial.csv'):
        s = SB / region / rel
        if s.is_file():
            shutil.copy2(s, out / s.name)
            copied.append(s.name)
    (out / 'REBUILD_INFO.json').write_text(json.dumps(dict(
        region=region,
        review_policy=policy,
        rebuilt_at=datetime.now(timezone.utc).isoformat(),
        note=('Rebuilt from OSM with the current pipeline. NOT a reviewed model: '
              'Step 3 was answered automatically under the stated policy. '
              'Kept separate from the checked-in case in ../../{region}/.'),
    ), indent=2), encoding='utf-8')
    return copied


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        region, _, policy = arg.partition('=')
        copied = preserve(region, policy or 'skip')
        if copied is None:
            print(f'{region}: sandbox not found')
        else:
            has_case = any(n.endswith('.m') for n in copied)
            print(f'{region}: {len(copied)} file(s) -> '
                  f'data/working_files/_rebuilt/{region}/'
                  f'{"  [case .m included]" if has_case else "  [no .m yet]"}')


if __name__ == '__main__':
    main()
