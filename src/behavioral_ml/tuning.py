"""Development-only selection of shared P2 hyperparameters with Optuna.

Personal enrollment remains genuine-only. Labeled development probes select
population-level settings; evaluation users enroll independently afterwards.
Identity separation is conditional on KMT's inferred typist candidates.
"""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
from pyod.models.knn import KNN
from sklearn.svm import OneClassSVM

from . import evaluation, tracking, train
from .experiment import _pipeline, estimator_parameters, write_results


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _save_contract(path, value):
    """A run directory cannot silently change its protocol or input contract."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing {path.name} differs; use a new output directory for a new study")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def make_identity_split(features, typist_candidates, n_enroll=5,
                        evaluation_fraction=0.3, random_state=42):
    """Predeclare people and exact probe rows without inspecting model scores.

    Keep a probe only if its owner and every candidate typist are in the same
    partition. Retain genuine enrollment independently for every claimed user.
    Return a JSON manifest and an attempt-level inclusion/exclusion audit.
    """
    if not 0 < evaluation_fraction < 1:
        raise ValueError("evaluation_fraction must be between zero and one")
    enrollments, probes = train.split_enroll_probe(features, n_enroll)
    candidates = typist_candidates.reindex(features.index)
    if candidates.isna().any():
        raise ValueError("Typist candidates must cover every attempt; use () for unknown")
    users = np.sort(features.index.get_level_values("User ID").unique())
    n_evaluation = int(np.ceil(len(users) * evaluation_fraction))
    if min(n_evaluation, len(users) - n_evaluation) < 2:
        raise ValueError("Each partition needs at least two people to retain impostor probes")
    shuffled = np.random.default_rng(random_state).permutation(users)
    evaluation_users = set(shuffled[:n_evaluation])
    development_users = set(shuffled[n_evaluation:])
    enrollment_index = set(enrollments.index)
    rows = []
    retained = {"development": [], "evaluation": []}
    for identity in features.sort_index().index:
        owner, genuine, test = identity
        partition = "development" if owner in development_users else "evaluation"
        people = development_users if partition == "development" else evaluation_users
        possible = tuple(int(person) for person in candidates.loc[identity])
        if genuine and set(possible) != {int(owner)}:
            raise ValueError("A genuine attempt must identify its owner as the typist")
        if identity in enrollment_index:
            role, reason, keep = "enrollment", "genuine_enrollment", True
        else:
            role = "probe"
            keep = bool(possible) and set(possible) <= people
            reason = "retained" if keep else ("unknown_typist" if not possible else "cross_split_typist")
            if keep:
                retained[partition].append([int(owner), bool(genuine), int(test)])
        rows.append({"User ID": int(owner), "Is Genuine": bool(genuine), "Test ID": int(test),
                     "partition": partition, "role": role, "retained": keep,
                     "reason": reason, "typist_candidates": json.dumps(possible)})
    audit = pd.DataFrame(rows)
    coverage = {}
    for partition, indices in retained.items():
        labels = [row[1] for row in indices]
        both = sum(len({row[1] for row in indices if row[0] == user}) == 2
                   for user in (development_users if partition == "development" else evaluation_users))
        if not labels or not all(label in labels for label in [False, True]) or not both:
            raise ValueError(f"No two-class probes remain in {partition}; revise the declared split protocol")
        owned = probes.index.get_level_values("User ID").isin(
            development_users if partition == "development" else evaluation_users)
        coverage[partition] = {"n_candidate_probes": int(owned.sum()), "n_probes": len(labels),
                               "n_genuine": sum(labels), "n_impostor": len(labels) - sum(labels),
                               "n_users_with_both_classes": int(both),
                               "n_excluded_probes": int(owned.sum()) - len(labels)}
    manifest = {
        "version": 1, "n_enroll": int(n_enroll), "random_state": int(random_state),
        "evaluation_fraction": evaluation_fraction,
        "identity_policy": "owner and all plausible typists within partition; unknown excluded",
        "attribution_assumption": "all genuine sessions ending within 3600s before an impostor attempt",
        "development_users": sorted(int(user) for user in development_users),
        "evaluation_users": sorted(int(user) for user in evaluation_users),
        "development_probes": retained["development"], "evaluation_probes": retained["evaluation"],
        "coverage": coverage,
    }
    return manifest, audit


def _partition_features(features, split, phase):
    return features.loc[features.index.get_level_values("User ID").isin(split[f"{phase}_users"])].copy()


def score_partition(detector, features, split, phase, fitted_models=None):
    """Score one partition; its exact probe mask is fixed before optimization."""
    if set(features.index.get_level_values("User ID")) != set(split[f"{phase}_users"]):
        raise ValueError("Scoring input must contain only the declared partition's users")
    scores = train.enroll_and_score(detector, features, split["n_enroll"], fitted_models=fitted_models)
    index = pd.MultiIndex.from_tuples([tuple(row) for row in split[f"{phase}_probes"]], names=train.TEST_LEVEL)
    scores = scores.loc[index].sort_index()
    scores.attrs.update(phase=phase, identity_policy=split["identity_policy"],
                        **split["coverage"][phase])
    return scores


def build_tuned_detector(name, params):
    """Resolve only the declared tunable methods; preprocessing stays fixed."""
    if name == "knn":
        return _pipeline(train.PyODNoveltyAdapter(KNN(**params)))
    if name == "ocsvm":
        return _pipeline(OneClassSVM(kernel="rbf", **params))
    raise ValueError("Tuning supports knn and ocsvm")


def _suggest_params(trial, name, n_enroll):
    if name == "knn":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, n_enroll - 1),
                "method": trial.suggest_categorical("method", ["largest", "mean", "median"]),
                "metric": trial.suggest_categorical("metric", ["manhattan", "euclidean"])}
    return {"nu": trial.suggest_float("nu", .05, .5),
            "gamma": trial.suggest_float("gamma", 1e-3, 1., log=True)}


def _input_contract(feature_sets, split, audit, detector_names, n_trials, random_state):
    feature_hashes = {}
    for name, features in feature_sets.items():
        values = features.sort_index().astype(float)
        digest = hashlib.sha256(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
        digest.update(json.dumps(list(values.columns)).encode())
        feature_hashes[name] = digest.hexdigest()
    root = Path(__file__).parent
    return {
        "version": 1, "split_sha256": _json_digest(split),
        "typist_candidates_sha256": hashlib.sha256(audit.to_csv(index=False).encode()).hexdigest(),
        "feature_sha256": feature_hashes, "detectors": list(detector_names),
        "n_trials": int(n_trials), "random_state": int(random_state),
        "objective": "maximize mean per-user development ROC AUC; omit one-class users",
        "preprocessing": "enrollment-only median imputation (keep empty) and StandardScaler",
        "search_space": {"knn": {"n_neighbors": [1, split["n_enroll"] - 1],
                                  "method": ["largest", "mean", "median"],
                                  "metric": ["manhattan", "euclidean"]},
                         "ocsvm": {"nu": [.05, .5], "gamma_log": [1e-3, 1.]}},
        "versions": {name: version(name) for name in ["optuna", "pyod", "scikit-learn", "numpy", "pandas"]},
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(root.rglob("*.py"))},
    }


def run_tuning(feature_sets, typist_candidates, output_dir, n_enroll=5, n_trials=30,
               evaluation_fraction=.3, random_state=42, detector_names=("knn", "ocsvm"),
               tracking_uri=None, experiment_name="kmt", data_dir=None):
    """Tune shared settings, freeze every selection, then evaluate once.

    SQLite retains completed trials after interruption. Once selection.json
    exists, reruns use the frozen selections and cannot append trials. A changed
    contract requires another output directory. All methods/features are
    predeclared; the overall winner is selected using development AUC only.
    """
    if not feature_sets or not detector_names or len(set(detector_names)) != len(detector_names):
        raise ValueError("Provide feature sets and unique detector names")
    if set(detector_names) - {"knn", "ocsvm"}:
        raise ValueError("Tuning supports knn and ocsvm")
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if n_enroll < 2:
        raise ValueError("Tuning needs at least two genuine enrollment attempts")
    reference = next(iter(feature_sets.values()))
    for features in feature_sets.values():
        if not features.index.sort_values().equals(reference.index.sort_values()):
            raise ValueError("Feature sets must contain identical attempt identities")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split, audit = make_identity_split(reference, typist_candidates, n_enroll, evaluation_fraction, random_state)
    _save_contract(output_dir / "split.json", split)
    contract = _input_contract(feature_sets, split, audit, detector_names, n_trials, random_state)
    _save_contract(output_dir / "study_contract.json", contract)
    audit.to_csv(output_dir / "split_audit.csv", index=False)
    client, experiment_id = tracking.create_client(output_dir, tracking_uri, experiment_name)
    selection_path = output_dir / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["contract_sha256"] != _json_digest(contract):
            raise ValueError("Frozen selection does not match the study contract")
    else:
        choices = {}
        for detector_name in detector_names:
            for feature_name, features in feature_sets.items():
                name = f"p2.{detector_name}.{feature_name}"
                development = _partition_features(features, split, "development")
                if detector_name == "knn":
                    grid = {"n_neighbors": list(range(1, n_enroll)),
                            "method": ["largest", "mean", "median"],
                            "metric": ["manhattan", "euclidean"]}
                    sampler = optuna.samplers.GridSampler(grid, seed=random_state)
                    budget = min(n_trials, (n_enroll - 1) * 6)
                else:
                    sampler = optuna.samplers.TPESampler(seed=random_state)
                    budget = n_trials
                study = optuna.create_study(
                    study_name=f"{name}.{_json_digest(contract)[:16]}",
                    storage=f"sqlite:///{output_dir / 'optuna.db'}", load_if_exists=True,
                    sampler=sampler, pruner=optuna.pruners.NopPruner(), direction="maximize")
                with tracking.run(client, experiment_id, name, tags={"phase": "tuning", "study": study.study_name}) as parent_id:
                    for artifact in ["split.json", "split_audit.csv", "study_contract.json"]:
                        client.log_artifact(parent_id, str(output_dir / artifact))

                    def objective(trial):
                        params = _suggest_params(trial, detector_name, n_enroll)
                        detector = build_tuned_detector(detector_name, params)
                        with tracking.run(client, experiment_id, f"trial-{trial.number:03d}", parent_id,
                                          {"phase": "development", "experiment": name}) as run_id:
                            trial.set_user_attr("mlflow_run_id", run_id)
                            scores = score_partition(detector, development, split, "development")
                            scores.attrs.update(feature_set=feature_name, feature_columns=list(features),
                                                random_state=random_state, estimator=repr(detector),
                                                estimator_params=estimator_parameters(detector))
                            value = float(evaluation.per_user_metrics(scores)["auc"].mean())
                            tracking.log_score_run(client, run_id, scores, params={"trial_number": trial.number})
                            client.log_metric(run_id, "development_mean_user_auc", value)
                        return value

                    completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
                    study.optimize(objective, n_trials=max(0, budget - completed), n_jobs=1)
                    table_path = output_dir / f"{name}.trials.csv"
                    study.trials_dataframe().to_csv(table_path, index=False)
                    client.log_artifact(parent_id, str(table_path))
                    client.log_metric(parent_id, "best_development_mean_user_auc", study.best_value)
                    choices[name] = {"detector": detector_name, "feature_set": feature_name,
                                     "params": study.best_params, "development_mean_user_auc": study.best_value,
                                     "study_name": study.study_name, "best_trial": study.best_trial.number,
                                     "mlflow_parent_run_id": parent_id,
                                     "best_trial_run_id": study.best_trial.user_attrs["mlflow_run_id"]}
        winner = sorted(choices, key=lambda key: (-choices[key]["development_mean_user_auc"], key))[0]
        selection = {"contract_sha256": _json_digest(contract), "selected_experiment": winner,
                     "selection_metric": contract["objective"], "configurations": choices,
                     "threshold_policy": "descriptive only; no independent deployment calibration"}
        # Persist the entire selection BEFORE opening any evaluation probe scores.
        _save_contract(selection_path, selection)
    experiments = {}
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    for name, choice in selection["configurations"].items():
        features = feature_sets[choice["feature_set"]]
        detector = build_tuned_detector(choice["detector"], choice["params"])
        models = {}
        scores = score_partition(detector, _partition_features(features, split, "evaluation"),
                                 split, "evaluation", fitted_models=models)
        scores.attrs.update(feature_set=choice["feature_set"], feature_columns=list(features),
                            random_state=random_state, estimator=repr(detector),
                            estimator_params=estimator_parameters(detector),
                            split_sha256=contract["split_sha256"],
                            selected_on_development=name == selection["selected_experiment"])
        experiments[name] = scores
        joblib.dump(models, model_dir / f"{name}.joblib")
    evaluation_dir = output_dir / "evaluation"
    write_results(experiments, evaluation_dir, data_dir)
    extra = [output_dir / filename for filename in ["split.json", "split_audit.csv", "study_contract.json", "selection.json"]]
    extra.extend(sorted(model_dir.glob("*.joblib")))
    parent_id = tracking.log_experiments(experiments, evaluation_dir, tracking_uri=client.tracking_uri,
                                        experiment_name=experiment_name, run_name="frozen-evaluation",
                                        phase="evaluation", extra_artifacts=extra)
    client.set_tag(parent_id, "selected_experiment", selection["selected_experiment"])
    return experiments, selection
