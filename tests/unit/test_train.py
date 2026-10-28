import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pyod.models.knn import KNN

from behavioral_ml import train


@pytest.fixture
def features():
    rows, values = [], []
    for user in [10, 20, 30, 40]:
        for genuine in [False, True]:
            for test in range(1, 6):
                rows.append((user, genuine, test))
                values.append([user + test / 10 + (0 if genuine else 5), test % 2])
    index = pd.MultiIndex.from_tuples(rows, names=train.TEST_LEVEL)
    return pd.DataFrame(values, index=index, columns=["timing", "habit"])


def test_enrollment_is_chronological_even_when_input_is_shuffled(features):
    enroll, probes = train.split_enroll_probe(features.sample(frac=1, random_state=8), 2)
    assert len(enroll) == 8
    assert len(probes) == 32
    assert enroll.index.get_level_values("Is Genuine").all()
    assert set(enroll.index.get_level_values("Test ID")) == {1, 2}
    assert enroll.index.intersection(probes.index).empty
    with pytest.raises(ValueError, match="both enrollment and probing"):
        train.split_enroll_probe(features, 5)


def test_comparison_vectors_are_absolute_and_only_use_enrollment(features):
    absolute = train.build_comparison_vectors(features, 2)
    signed = train.build_comparison_vectors(features, 2, absolute=False)
    np.testing.assert_allclose(absolute, signed.abs())
    assert absolute.loc[(10, True, 3), "timing"] == pytest.approx(.15)
    changed = features.copy()
    changed.loc[(10, True, 5), "timing"] = 1_000_000
    unchanged = train.build_comparison_vectors(changed, 2)
    pd.testing.assert_series_equal(absolute.loc[(10, True, 3)], unchanged.loc[(10, True, 3)])


@pytest.mark.parametrize("classifier", [SVC(), RandomForestClassifier(n_estimators=5, random_state=42)])
def test_per_user_cv_supports_margins_and_probabilities_with_indexed_scores(features, classifier):
    shuffled = features.sample(frac=1, random_state=3)
    model = make_pipeline(StandardScaler(), classifier)
    scores = train.cross_val_predict_per_user(model, shuffled, StratifiedKFold(2, shuffle=True, random_state=42))
    assert scores.index.equals(features.index.sort_values())
    assert np.isfinite(scores).all()
    assert len(scores.attrs["folds"]) == len(features)


def test_detector_cloning_and_preprocessing_use_enrollment_only(features):
    detector = train.TemplateDetector()
    scores = train.enroll_and_score(detector, features, 2)
    assert not hasattr(detector, "center_")
    changed = features.copy()
    changed.loc[(10, False, 5), "timing"] = 1_000_000
    changed_scores = train.enroll_and_score(detector, changed, 2)
    unaffected = scores.index.drop((10, False, 5))
    np.testing.assert_allclose(scores.loc[unaffected], changed_scores.loc[unaffected])


def test_template_scale_handles_zero_spread_and_all_missing_columns():
    detector = train.TemplateDetector(center="mean", scale="mad").fit(
        np.array([[1., 5., np.nan], [3., 5., np.nan]]))
    np.testing.assert_allclose(detector.score_samples([[4., 5., np.nan]]), [-2.])


def test_pyod_adapter_preserves_distance_direction_cloning_and_probe_independence():
    underlying = KNN(n_neighbors=1)
    adapter = train.PyODNoveltyAdapter(underlying).fit([[0.], [1.], [2.], [3.], [4.]])
    np.testing.assert_allclose(adapter.decision_function([[2.], [10.]]), [0., -6.])
    np.testing.assert_allclose(adapter.decision_function([[2.]]), [0.])
    assert not hasattr(underlying, "decision_scores_")


def test_pyod_pipeline_uses_enrollment_only_and_keeps_attempt_identity(features):
    detector = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                             train.PyODNoveltyAdapter(KNN(n_neighbors=1)))
    scores = train.enroll_and_score(detector, features, n_enroll=2)
    changed = features.copy()
    changed.loc[(10, False, 5), "timing"] = 1_000_000
    rescored = train.enroll_and_score(detector, changed, n_enroll=2)
    index = scores.index.drop((10, False, 5))
    pd.testing.assert_series_equal(scores.loc[index], rescored.loc[index])
    assert scores.index.equals(train.split_enroll_probe(features, 2)[1].index)


class IdentityAuditClassifier(BaseEstimator):
    """Fails if any plausible probe typist was present during model fitting."""
    candidates = None

    def fit(self, X, y):
        self.people_ = set(X.index.get_level_values("User ID"))
        for index in X.index:
            assert set(self.candidates.loc[index]) <= self.people_
        return self

    def decision_function(self, X):
        assert not (self.people_ & set(X.index.get_level_values("User ID")))
        for index in X.index:
            assert not (self.people_ & set(self.candidates.loc[index]))
        return -X.iloc[:, 0].to_numpy()


def test_global_cv_excludes_cross_fold_and_ambiguous_typists(features):
    # GroupKFold(2) holds {20, 40} and {10, 30} in opposite folds.
    same_fold = {10: 30, 30: 10, 20: 40, 40: 20}
    candidates = pd.Series([
        (u,) if genuine else (same_fold[u],)
        for u, genuine, _ in features.index], index=features.index)
    candidates.loc[(10, False, 1)] = (20, 30)
    candidates.loc[(20, False, 1)] = ()
    IdentityAuditClassifier.candidates = candidates
    scores = train.cross_val_predict_global(IdentityAuditClassifier(), features, GroupKFold(2),
                                            n_enroll=2, typist_candidates=candidates)
    assert (10, False, 1) not in scores.index
    assert (20, False, 1) not in scores.index
    assert scores.attrs["n_excluded_probes"] == 2
    assert len(scores) == 30
    with pytest.raises(ValueError, match="typist_candidates"):
        train.cross_val_predict_global(SVC(), features, GroupKFold(2), n_enroll=2)
