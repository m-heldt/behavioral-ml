import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_curve

from behavioral_ml import evaluation


def test_eer_keeps_crossing_removed_by_curve_thinning():
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1]
    scores = np.arange(10, 0, -1)
    far, tar, thresholds = roc_curve(labels, scores)
    assert evaluation._eer_from_curve(far, 1 - tar, thresholds)[0] == pytest.approx(.1)
    assert evaluation.eer_score(labels, scores) == pytest.approx(.2)
    assert evaluation.verification_report(labels, scores)["eer"] == pytest.approx(.2)


def test_far_target_is_a_conservative_attainable_point_and_handles_ties():
    labels = [1, 1, 0, 0]
    scores = [.9, .5, .5, .1]
    threshold = evaluation.threshold_at_far(labels, scores, far=.01)
    assert threshold == .9
    assert evaluation.metrics_at_threshold(labels, scores, threshold) == {"far": 0., "frr": .5}
    assert evaluation.frr_at_far(labels, scores, .01) == .5
    assert evaluation.frr_at_far(labels, scores, .5) == 0.


@pytest.mark.parametrize("labels,scores", [([1, 1], [0, 1]), ([0, 1], [0, np.nan]), ([0, 1], [0])])
def test_invalid_verification_inputs_fail_explicitly(labels, scores):
    with pytest.raises(ValueError):
        evaluation.eer_score(labels, scores)


def test_per_user_reporting_preserves_missing_class_and_pooled_distinction():
    index = pd.MultiIndex.from_tuples([(10, True, 1), (10, False, 1), (20, True, 1)],
                                     names=["User ID", "Is Genuine", "Test ID"])
    scores = pd.Series([.9, .1, -.5], index=index)
    per_user = evaluation.per_user_metrics(scores)
    assert len(per_user) == 2
    assert per_user.loc[10, "eer"] == 0
    assert np.isnan(per_user.loc[20, "eer"])
    report = evaluation.summarize(scores)
    assert report["n_users_with_both_classes"] == 1
    assert report["per_user_eer_mean"] == 0
    assert report["pooled_auc"] == .5


def test_plot_functions_compose_on_existing_axes():
    import matplotlib.pyplot as plt
    report = evaluation.verification_report([0, 1, 0, 1], [.1, .8, .4, .7])
    _, ax = plt.subplots()
    assert evaluation.plot_roc_curve(report, ax=ax, label="template") is ax
    assert evaluation.plot_det_curve(report, ax=ax, label="template") is ax
    plt.close("all")
