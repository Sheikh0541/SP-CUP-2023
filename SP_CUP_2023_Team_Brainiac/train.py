import argparse
import glob
from pathlib import Path

import joblib
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


def get_data(dir: Path):
    fnc_pat = dir / "*/*/fnc.npy"
    fnc_pat = fnc_pat.as_posix()
    fnc_paths = glob.glob(fnc_pat)

    fnc_data = [np.load(path).ravel() for path in fnc_paths]
    labels = [path.split("/")[-3] == "BP" for path in fnc_paths]
    X = np.vstack(fnc_data, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="Training data directory")
    parser.add_argument("model_dir", help="Output models directory")

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    model_dir = Path(args.model_dir)

    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = get_data(train_dir)

    outer_cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        inner_cv = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

        model = SVC(probability=True, random_state=42, kernel='poly')
        resample = SMOTETomek(random_state=42)
        pipeline = Pipeline(steps=[("r", resample), ("m", model)])

        param_grid = {
            "m__C": np.logspace(-3, 3, 7),
            "m__kernel": ['rbf', 'poly', 'sigmoid'],
            "m__gamma": np.logspace(-3, 3, 7),
        }

        clf = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, y_pred)
        print(f"Fold {fold}/5 Done!")
        print(f"AUC: {score}")
        print(f"Best params: {clf.best_params_}")
        print("")

        scores.append(score)
        joblib.dump(clf, model_dir / f"model_fold_{fold}.pkl")

    print(f"Average AUC: {np.mean(scores): .4f} (+/- {np.std(scores): .4f})")
