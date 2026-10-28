import numpy as np
import pandas as pd
import pytest

from behavioral_ml import experiment, train


def test_personal_matrix_keeps_probes_identical_across_modalities_and_detectors():
    index = pd.MultiIndex.from_tuples(
        [(user, genuine, test) for user in [10, 20]
         for genuine in [False, True] for test in range(1, 7)], names=train.TEST_LEVEL)
    values = np.array([[user + test / 10 + (0 if genuine else 5), test % 2]
                       for user, genuine, test in index])
    keys = pd.DataFrame(values, index=index, columns=["timing", "habit"])
    mouse = keys.rename(columns={"timing": "speed", "habit": "clicks"}) * 2
    features = {"keys": keys, "mouse": mouse, "keys_mouse": keys.join(mouse)}
    results = experiment.run_personal_matrix(features, n_enroll=3)
    assert len(results) == 21
    probe_index = train.split_enroll_probe(keys, 3)[1].index
    for name, scores in results.items():
        assert scores.index.equals(probe_index)
        assert np.isfinite(scores).all()
        assert scores.attrs["feature_set"] == name.split(".")[2]
        assert "estimator_params" in scores.attrs
    knn = results["p2.knn.keys"]
    assert knn.loc[(10, True, 4)] > knn.loc[(10, False, 4)]
    # Changing a probe cannot change another probe or the fitted enrollment.
    changed = keys.copy()
    changed.loc[(10, False, 6), "timing"] = 1e9
    changed_scores = train.enroll_and_score(experiment.personal_detectors(3)["knn"], changed, 3)
    unaffected = knn.index.drop((10, False, 6))
    np.testing.assert_allclose(knn.loc[unaffected], changed_scores.loc[unaffected])
    with pytest.raises(ValueError, match="identical attempt"):
        experiment.run_personal_matrix({"keys": keys, "mouse": mouse.iloc[:-1]}, 3)


def test_matrix_neighbor_counts_follow_enrollment_size():
    for n_enroll in [2, 3, 5]:
        detectors = experiment.personal_detectors(n_enroll)
        for name in ["knn", "lof"]:
            wrapped = detectors[name].steps[-1][1].detector
            assert 1 <= wrapped.n_neighbors < n_enroll
    with pytest.raises(ValueError, match="at least two"):
        experiment.personal_detectors(1)
