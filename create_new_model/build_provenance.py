"""Record the settings a region was built with, and refuse inconsistent reruns.

`config.py` is global and gets overwritten for each new region, so a finished
model carried no record of the MIN_TRANSMISSION_VOLTAGE, demand or export used
to produce it. That makes an existing case impossible to reproduce or audit, and
lets a half-finished build silently mix settings when config.py is edited between
steps.

Step 1 writes `build_config.json` into the region's working directory; later
steps verify against it and append themselves to the step log.
"""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

# Settings that change the produced model. Editing any of these invalidates
# outputs from earlier steps, so a mismatch is an error rather than a warning.
SETTINGS = (
    "region", "COUNTRY_ISO3", "REGION_BBOX", "SHAPE_FILE_NAME",
    "MIN_TRANSMISSION_VOLTAGE_V", "COUNTRY_DEMAND_MW", "NATIONAL_EXPORT_MW",
    "LOAD_MODEL_METHOD", "INFER_TRANSFORMERS_FROM_VOLTAGE", "FUEL_COST_BASIS",
    "PLANT_NAME_SEARCH_RADIUS_M",
)

FILENAME = "build_config.json"


def snapshot(settings):
    """Read the reproducibility-relevant settings out of a config module."""
    out = {}
    for name in SETTINGS:
        if not hasattr(settings, name):
            continue
        value = getattr(settings, name)
        out[name] = list(value) if isinstance(value, tuple) else value
    return out


def file_digest(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def differences(recorded, current):
    """Setting names whose recorded value differs from the current one."""
    return sorted(name for name in set(recorded) | set(current)
                  if recorded.get(name) != current.get(name))


def load(work_dir):
    path = Path(work_dir) / FILENAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def record(settings, work_dir, step, inputs=()):
    """Create or update the build record, appending this step to the log."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    current = snapshot(settings)
    existing = load(work_dir)
    if existing is None:
        existing = dict(settings=current, steps=[], inputs={})
    else:
        changed = differences(existing.get("settings", {}), current)
        if changed:
            raise ValueError(
                f"config.py changed since this region was started: {', '.join(changed)}. "
                f"Earlier outputs in {work_dir} were built with different settings. "
                f"Restore the recorded values in {work_dir / FILENAME}, or delete that "
                "directory and rebuild the region from Step 1.")
    for path in inputs:
        digest = file_digest(path)
        if digest is not None:
            existing.setdefault("inputs", {})[Path(path).name] = digest
    existing["settings"] = current
    existing.setdefault("steps", []).append(
        dict(step=step, at=datetime.now(timezone.utc).isoformat()))
    (work_dir / FILENAME).write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return existing


def verify(settings, work_dir, step=None):
    """Raise when the current config contradicts the recorded build.

    A missing record is not an error: every region built before this file
    existed lacks one, and refusing to run would block rework on them. In that
    case the current settings are adopted and recorded, with a warning that they
    were never checked against Step 1.
    """
    existing = load(work_dir)
    if existing is None:
        print(f"  WARNING: no {FILENAME} in {work_dir}; this region predates build "
              "provenance. Recording the current config.py settings, which have NOT "
              "been verified against the settings its existing tables were built with.")
        return record(settings, work_dir, step or "adopted-existing")
    changed = differences(existing.get("settings", {}), snapshot(settings))
    if changed:
        raise ValueError(
            f"config.py no longer matches this region's build: {', '.join(changed)}. "
            f"Recorded settings are in {Path(work_dir) / FILENAME}. Restore them, or "
            "rebuild the region from Step 1.")
    if step is not None:
        return record(settings, work_dir, step)
    return existing
