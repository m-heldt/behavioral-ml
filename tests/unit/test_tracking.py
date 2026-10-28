import json
from pathlib import Path

from mlflow import MlflowClient
import pandas as pd

from behavioral_ml.experiment import write_results
from behavioral_ml.tracking import log_experiments


def test_tracking_records_configuration_scores_folds_and_common_coverage(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [(10, False, 1), (10, True, 3), (20, False, 1), (20, True, 3)],
        names=["User ID", "Is Genuine", "Test ID"])
    scores = pd.Series([.1, .9, .2, .8], index=index, name="score")
    scores.attrs.update(protocol="p2", feature_set="keys", n_enroll=2,
                        estimator_params={"metric": "manhattan"})
    restricted = scores.iloc[:3].copy()
    restricted.attrs["folds"] = {identity: 0 for identity in restricted.index}
    experiments = {"p2.template_l1": scores, "p2.global_logreg": restricted}
    write_results(experiments, tmp_path)
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    parent_id = log_experiments(experiments, tmp_path, uri, "test-kmt")
    client = MlflowClient(tracking_uri=uri)
    parent = client.get_run(parent_id)
    children = client.search_runs([parent.info.experiment_id],
                                 filter_string=f"tags.mlflow.parentRunId = '{parent_id}'")
    assert parent.info.status == "FINISHED"
    assert len(children) == 2
    child = next(item for item in children if item.data.tags["experiment"] == "p2.template_l1")
    assert child.data.params["n_enroll"] == "2"
    assert child.data.params["estimator.metric"] == "manhattan"
    assert child.data.metrics["all_probes.n_probes"] == 4
    assert child.data.metrics["common_probes.n_probes"] == 3
    assert child.data.metrics["common_probes.n_users_with_both_classes"] == 1
    path = client.download_artifacts(child.info.run_id, "configuration.json", str(tmp_path))
    assert json.loads(Path(path).read_text())["estimator_params"] == {"metric": "manhattan"}
    global_run = next(item for item in children if item.data.tags["experiment"] == "p2.global_logreg")
    path = client.download_artifacts(global_run.info.run_id, "predictions.csv", str(tmp_path))
    predictions = pd.read_csv(path)
    assert predictions["fold"].tolist() == [0, 0, 0]
    assert predictions["score"].tolist() == restricted.tolist()
    assert {item.path for item in client.list_artifacts(child.info.run_id)} >= {
        "all_probes_det.png", "common_probes_roc.png", "per_user_all_probes.csv"}
