"""Repeatable baseline experiments used by the notebook and command-line entry point."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM, SVC

from . import evaluation, train
from .etl import infer_typist_candidates, load_data
from .features import build_feature_sets


def _pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), estimator)


def run_baselines(feature_sets, typist_candidates, n_enroll=5, n_splits=5, random_state=42):
    """Fixed baseline settings, without selecting hyperparameters on the results."""
    keys = feature_sets["keys"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    experiments = {
        "p1.logreg": train.cross_val_predict_per_user(
            _pipeline(LogisticRegression(C=1, max_iter=1000)), keys, cv)
    }
    detectors = {
        "p2.template_l1": train.TemplateDetector(center="median", scale="std"),
        "p2.scaled_l1": train.TemplateDetector(center="mean", scale="mad"),
        "p2.ocsvm": _pipeline(OneClassSVM(kernel="rbf", gamma="scale")),
        "p2.iforest": _pipeline(IsolationForest(n_estimators=200, random_state=random_state)),
    }
    for name, detector in detectors.items():
        experiments[name] = train.enroll_and_score(detector, keys, n_enroll)
        experiments[name].attrs["estimator"] = repr(detector)
    global_models = {
        "p2.global_svc": _pipeline(SVC(kernel="rbf", gamma="scale")),
        "p2.global_logreg": _pipeline(LogisticRegression(C=1, max_iter=1000)),
    }
    for name, model in global_models.items():
        group_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        experiments[name] = train.cross_val_predict_global(
            model, keys, group_cv, n_enroll=n_enroll, typist_candidates=typist_candidates)
        experiments[name].attrs["estimator"] = repr(model)
    for scores in experiments.values():
        scores.attrs.update(feature_set="keys", feature_columns=list(keys),
                            n_splits=n_splits, random_state=random_state)
    experiments["p1.logreg"].attrs["estimator"] = repr(_pipeline(LogisticRegression(C=1, max_iter=1000)))
    return experiments


def common_probe_scores(experiments):
    """Restrict every method, including P1, to the same scored P2 probe rows."""
    common = None
    for name, scores in experiments.items():
        if name.startswith("p2."):
            common = scores.index if common is None else common.intersection(scores.index)
    if common is None or common.empty:
        raise ValueError("At least one P2 experiment with common probes is required")
    return {name: scores.loc[common] for name, scores in experiments.items()}


def results_table(experiments):
    return pd.DataFrame({name: evaluation.summarize(scores)
                         for name, scores in experiments.items()}).T.rename_axis("experiment")


def write_results(experiments, output_dir, data_dir=None):
    """Save indexed predictions, fold provenance, per-user metrics, and settings."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = common_probe_scores(experiments)
    results_table(experiments).to_csv(output_dir / "all_probes_metrics.csv")
    results_table(common).to_csv(output_dir / "common_probes_metrics.csv")
    predictions, per_user = [], []
    configuration = {}
    for name, scores in experiments.items():
        frame = scores.to_frame()
        folds = scores.attrs.get("folds", {})
        frame["fold"] = [folds.get(index, -1) for index in scores.index]
        frame["experiment"] = name
        predictions.append(frame.reset_index())
        per_user.append(evaluation.per_user_metrics(common[name]).assign(experiment=name).reset_index())
        configuration[name] = {k: v for k, v in scores.attrs.items() if k != "folds"}
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(per_user, ignore_index=True).to_csv(output_dir / "per_user_common_probes.csv", index=False)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "git_revision": revision.stdout.strip(),
        "source_sha256": {str(path.relative_to(root)): digest(path)
                          for path in sorted((root / "src").rglob("*.py"))},
        "experiments": configuration,
        "attribution_assumption": "all genuine sessions ending within 3600s before an impostor attempt",
        "threshold_policy": "EER and FRR-at-FAR are descriptive evaluation-curve statistics",
    }
    if data_dir is not None:
        provenance["dataset_sha256"] = {path.name: digest(path) for path in sorted(Path(data_dir).glob("raw_kmt_user_*.json"))}
    (output_dir / "configuration.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run fixed KMT behavioral-verification baselines")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("KMT_DATA_DIR", "data/raw_kmt_dataset")))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-enroll", type=int, default=5)
    args = parser.parse_args(argv)
    mouse_events, key_events = load_data(args.data_dir)
    feature_sets = build_feature_sets(mouse_events, key_events)
    candidates = infer_typist_candidates(key_events)
    experiments = run_baselines(feature_sets, candidates, n_enroll=args.n_enroll)
    write_results(experiments, args.output_dir, args.data_dir)
    print(results_table(common_probe_scores(experiments)).to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
