import argparse
import glob
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def get_data(dir: Path):
    fnc_pat = dir / "*/fnc.npy"
    fnc_pat = fnc_pat.as_posix()
    fnc_paths = glob.glob(fnc_pat)

    fnc_data = [np.load(path).ravel() for path in fnc_paths]
    sub_id = [path.split("/")[-2] for path in fnc_paths]
    X = np.vstack(fnc_data, dtype=np.float32)
    return X, sub_id


def get_models(dir: Path) -> List[BaseEstimator]:
    model_pat = dir / "*.pkl"
    model_pat = model_pat.as_posix()
    model_paths = glob.glob(model_pat)

    return [joblib.load(path) for path in model_paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_dir", help="Test data directory")
    parser.add_argument("model_dir", help="Trained models directory")
    parser.add_argument("output_fp", help="Output file path")

    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)
    output_fp = Path(args.output_fp)

    X, sub_id = get_data(test_dir)
    models = get_models(model_dir)

    preds = [model.predict_proba(X)[:, 1] for model in models]

    pred = pd.DataFrame(
        {
            "ID": sub_id,
            "Predicted": np.mean(preds, axis=0),
        }
    )
    pred.sort_values(by='ID', inplace=True)
    pred.to_csv(output_fp, index=False)
