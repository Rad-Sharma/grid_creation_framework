"""Fit equation (1) using complete, preprocessed NZ/GB metered training rows.

Example: python train_paper_load_model.py prepared_nz.csv prepared_gb.csv
The old feature tables are not sufficient: their industrial light columns are
placeholders. They must be regenerated using paper feature extraction first.
This creates a new refit, not a claim to recover the paper's exact fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from paper_load_model import FEATURES, FEATURE_CONTRACT, PUBLISHED_COEFFICIENTS, feature_matrix


def train(paths, output):
    tables = [pd.read_csv(p) for p in paths]
    frame = pd.concat(tables, ignore_index=True)
    x = feature_matrix(frame)
    if "avg_mw" not in frame:
        raise ValueError("Training rows require metered avg_mw")
    y = frame.avg_mw.to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Invalid metered training demand")
    # Refuse an apparently complete table that cannot identify all six terms.
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), x])) < 7:
        raise ValueError("Training features cannot identify all six coefficients and the intercept")
    model = HuberRegressor(epsilon=1.1, alpha=0., max_iter=10000, tol=1e-9)
    model.fit(x, np.log1p(y))
    if model.n_iter_ >= model.max_iter:
        raise RuntimeError("Huber training did not converge")
    artifact = dict(model_type="huber", epsilon=1.1, features=FEATURES,
                    feature_contract=FEATURE_CONTRACT, intercept=float(model.intercept_),
                    coefficients=model.coef_.tolist(), training_n=len(frame),
                    published_coefficients=PUBLISHED_COEFFICIENTS,
                    provenance={"kind": "new_refit_not_published_artifact", "alpha": 0.,
                                "note": ("the paper omits intercept, regularization and "
                                         "the full GB correction/aggregation recipe."),
                                "inputs": [{"file": Path(p).name, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                                           for p in paths]})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/bus_data/models/model_huber_paper.json")
    args = parser.parse_args()
    train(args.training_csv, args.output)
