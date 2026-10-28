"""Repeatable baseline experiments used by the notebook and command-line entry point."""

import argparse
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import sklearn
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
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


def estimator_parameters(estimator):
    """JSON-safe resolved parameters, including pipeline preprocessing."""
    return json.loads(json.dumps(estimator.get_params(deep=True), default=str))


def personal_detectors(n_enroll=5, random_state=42):
    """Seven fixed baselines; neighbor counts respect enrollment self-exclusion."""
    if not isinstance(n_enroll, (int, np.integer)) or isinstance(n_enroll, bool) or n_enroll < 2:
        raise ValueError("The personal detector matrix needs at least two enrollment attempts")
    return {
        "template_l1": train.TemplateDetector(center="median", scale="std"),
        "scaled_l1": train.TemplateDetector(center="mean", scale="mad"),
        "ocsvm": _pipeline(OneClassSVM(kernel="rbf", gamma="scale")),
        "iforest": _pipeline(IsolationForest(n_estimators=200, random_state=random_state)),
        "knn": _pipeline(train.PyODNoveltyAdapter(KNN(
            n_neighbors=min(3, n_enroll - 1), method="mean", metric="manhattan"))),
        "lof": _pipeline(train.PyODNoveltyAdapter(LOF(
            n_neighbors=min(2, n_enroll - 1), novelty=True))),
        "hbos": _pipeline(train.PyODNoveltyAdapter(HBOS(n_bins=3))),
    }


def run_personal_matrix(feature_sets, n_enroll=5, random_state=42):
    """Score every detector × feature set on identical P2 probe identities."""
    if not feature_sets:
        raise ValueError("At least one feature set is required")
    reference = next(iter(feature_sets.values())).index.sort_values()
    detectors = personal_detectors(n_enroll, random_state)
    experiments = {}
    for feature_name, features in feature_sets.items():
        if not features.index.sort_values().equals(reference):
            raise ValueError("Feature sets must contain identical attempt identities")
        for detector_name, detector in detectors.items():
            scores = train.enroll_and_score(detector, features, n_enroll)
            scores.attrs.update(feature_set=feature_name, feature_columns=list(features),
                                random_state=random_state, estimator=repr(detector),
                                estimator_params=estimator_parameters(detector))
            experiments[f"p2.{detector_name}.{feature_name}"] = scores
    return experiments


def run_baselines(feature_sets, typist_candidates, n_enroll=5, n_splits=5, random_state=42):
    """Fixed personal matrix plus keyboard-only P1/global reference methods."""
    experiments = run_personal_matrix(feature_sets, n_enroll, random_state)
    keys = feature_sets["keys"]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logreg = _pipeline(LogisticRegression(C=1, max_iter=1000))
    references = {"p1.logreg": train.cross_val_predict_per_user(logreg, keys, cv)}
    references["p1.logreg"].attrs.update(estimator=repr(logreg),
                                        estimator_params=estimator_parameters(logreg))
    global_models = {
        "p2.global_svc": _pipeline(SVC(kernel="rbf", gamma="scale")),
        "p2.global_logreg": _pipeline(LogisticRegression(C=1, max_iter=1000)),
    }
    for name, model in global_models.items():
        group_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        references[name] = train.cross_val_predict_global(
            model, keys, group_cv, n_enroll=n_enroll, typist_candidates=typist_candidates)
        references[name].attrs.update(estimator=repr(model), estimator_params=estimator_parameters(model))
    for scores in references.values():
        scores.attrs.update(feature_set="keys", feature_columns=list(keys),
                            n_splits=n_splits, random_state=random_state)
    experiments.update(references)
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
    predictions, per_user, per_user_all = [], [], []
    configuration = {}
    for name, scores in experiments.items():
        frame = scores.to_frame()
        folds = scores.attrs.get("folds", {})
        frame["fold"] = [folds.get(index, -1) for index in scores.index]
        frame["experiment"] = name
        predictions.append(frame.reset_index())
        per_user.append(evaluation.per_user_metrics(common[name]).assign(experiment=name).reset_index())
        per_user_all.append(evaluation.per_user_metrics(scores).assign(experiment=name).reset_index())
        configuration[name] = {k: v for k, v in scores.attrs.items() if k != "folds"}
    pd.concat(predictions, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    pd.concat(per_user, ignore_index=True).to_csv(output_dir / "per_user_common_probes.csv", index=False)
    pd.concat(per_user_all, ignore_index=True).to_csv(output_dir / "per_user_all_probes.csv", index=False)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "pyod": version("pyod"), "mlflow": version("mlflow"), "optuna": version("optuna"),
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
    parser.add_argument("--tracking-uri", help="MLflow URI; default: output-dir/mlflow.db (SQLite)")
    parser.add_argument("--mlflow-experiment", default="kmt")
    parser.add_argument("--tune", action="store_true", help="Run development-only kNN/OCSVM tuning")
    parser.add_argument("--n-trials", type=int, default=30, help="Total trial budget per tuning study")
    parser.add_argument("--evaluation-fraction", type=float, default=.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-sets", nargs="+", choices=["keys", "mouse", "keys_mouse"],
                        default=["keys", "mouse", "keys_mouse"])
    parser.add_argument("--detectors", nargs="+", choices=["knn", "ocsvm"], default=["knn", "ocsvm"],
                        help="Methods to tune; fixed baseline runs always use seven personal methods")
    args = parser.parse_args(argv)
    mouse_events, key_events = load_data(args.data_dir)
    feature_sets = build_feature_sets(mouse_events, key_events)
    feature_sets = {name: feature_sets[name] for name in args.feature_sets}
    candidates = infer_typist_candidates(key_events)
    if args.tune:
        from .tuning import run_tuning
        experiments, selection = run_tuning(
            feature_sets, candidates, args.output_dir, n_enroll=args.n_enroll, n_trials=args.n_trials,
            evaluation_fraction=args.evaluation_fraction, random_state=args.seed,
            detector_names=tuple(args.detectors), tracking_uri=args.tracking_uri,
            experiment_name=args.mlflow_experiment, data_dir=args.data_dir)
        print(f"Selected on development: {selection['selected_experiment']}")
        print(results_table(experiments).to_string(float_format=lambda x: f"{x:.4f}"))
        return 0
    if "keys" not in feature_sets:
        parser.error("Baseline reference models require the keys feature set")
    experiments = run_baselines(feature_sets, candidates, n_enroll=args.n_enroll, random_state=args.seed)
    write_results(experiments, args.output_dir, args.data_dir)
    from .tracking import log_experiments
    run_id = log_experiments(experiments, args.output_dir, args.tracking_uri, args.mlflow_experiment)
    print(f"MLflow parent run: {run_id}")
    print(results_table(common_probe_scores(experiments)).to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
