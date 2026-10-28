"""Explicit MLflow logging for indexed biometric experiments.

Use the client API so notebook/global fluent tracking state is left alone.
One run represents a complete configuration, including all personal fits.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from mlflow import MlflowClient
import numpy as np

from . import evaluation


def create_client(output_dir, tracking_uri=None, experiment_name="kmt"):
    """Default to an absolute SQLite database and local artifacts directory."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{output_dir / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        location = (output_dir / "mlartifacts").as_uri() if uri.startswith("sqlite:") else None
        experiment_id = client.create_experiment(experiment_name, artifact_location=location)
    else:
        experiment_id = experiment.experiment_id
    return client, experiment_id


@contextmanager
def run(client, experiment_id, name, parent_id=None, tags=None):
    """Create a run and record failure as well as successful completion."""
    run_tags = {"mlflow.runName": name, **(tags or {})}
    if parent_id is not None:
        run_tags["mlflow.parentRunId"] = parent_id
    run_id = client.create_run(experiment_id, tags=run_tags).info.run_id
    try:
        yield run_id
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise
    else:
        client.set_terminated(run_id, status="FINISHED")


def log_score_run(client, run_id, scores, common_scores=None, params=None):
    """Log scores, finite metrics, class coverage, and plots to an existing run."""
    configuration = {key: value for key, value in scores.attrs.items() if key != "folds"}
    configuration.update(params or {})
    searchable = {**configuration, **{
        f"estimator.{key}": value
        for key, value in configuration.get("estimator_params", {}).items()}}
    for key, value in searchable.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            client.log_param(run_id, key, value)
    client.log_dict(run_id, configuration, "configuration.json")
    groups = {"all_probes": scores}
    if common_scores is not None:
        groups["common_probes"] = common_scores
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        predictions = scores.to_frame()
        folds = scores.attrs.get("folds", {})
        predictions["fold"] = [folds.get(index, -1) for index in scores.index]
        predictions.reset_index().to_csv(directory / "predictions.csv", index=False)
        for scope, values in groups.items():
            for metric, value in evaluation.summarize(values).items():
                if np.isfinite(value):
                    client.log_metric(run_id, f"{scope}.{metric}", float(value))
            users = evaluation.per_user_metrics(values)
            users.to_csv(directory / f"per_user_{scope}.csv")
            report = evaluation.verification_report(values.index.get_level_values("Is Genuine"), values)
            for name, plot in [("roc", evaluation.plot_roc_curve), ("det", evaluation.plot_det_curve)]:
                ax = plot(report)
                try:
                    ax.figure.savefig(directory / f"{scope}_{name}.png", bbox_inches="tight")
                finally:
                    plt.close(ax.figure)
        client.log_artifacts(run_id, str(directory))


def log_experiments(experiments, output_dir, tracking_uri=None,
                    experiment_name="kmt", run_name="baseline-matrix", phase="baseline",
                    extra_artifacts=()):
    """Log previously exported results, returning the parent MLflow run ID."""
    from .experiment import common_probe_scores

    output_dir = Path(output_dir).resolve()
    common = common_probe_scores(experiments)
    client, experiment_id = create_client(output_dir, tracking_uri, experiment_name)
    with run(client, experiment_id, run_name, tags={"phase": phase}) as parent_id:
        # Only upload the export files, never tracking databases or raw inputs.
        for name in ["configuration.json", "predictions.csv", "all_probes_metrics.csv",
                     "common_probes_metrics.csv", "per_user_common_probes.csv", "per_user_all_probes.csv"]:
            client.log_artifact(parent_id, str(output_dir / name))
        for path in extra_artifacts:
            client.log_artifact(parent_id, str(path))
        provenance = json.loads((output_dir / "configuration.json").read_text())
        client.set_tag(parent_id, "git_revision", provenance["git_revision"])
        for name, scores in experiments.items():
            tags = {"phase": phase, "experiment": name,
                    "score_direction": "higher_is_genuine"}
            with run(client, experiment_id, name, parent_id, tags) as child_id:
                log_score_run(client, child_id, scores, common[name])
    return parent_id
