"""Enrollment and out-of-fold scoring; metrics live in evaluation.py."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_is_fitted, validate_data

TEST_LEVEL = ["User ID", "Is Genuine", "Test ID"]


def _validate_features(features):
    if not isinstance(features.index, pd.MultiIndex) or list(features.index.names) != TEST_LEVEL:
        raise ValueError(f"Feature index must be {TEST_LEVEL}")
    if features.empty or not features.index.is_unique:
        raise ValueError("Features must be nonempty with unique attempt indices")
    if not features.index.get_level_values("Is Genuine").isin([False, True]).all():
        raise ValueError("Is Genuine must contain binary labels")


def get_labels(features):
    return features.index.get_level_values("Is Genuine").to_numpy(dtype=int)


def split_by_user(features):
    _validate_features(features)
    return {user: frame.copy() for user, frame in features.groupby(level="User ID", sort=True)}


def split_enroll_probe(features, n_enroll=5):
    """Select the first n genuine attempts by Test ID, keeping later probes."""
    if not isinstance(n_enroll, (int, np.integer)) or isinstance(n_enroll, bool) or n_enroll < 1:
        raise ValueError("n_enroll must be a positive integer")
    enrollments, probes = [], []
    for user, frame in split_by_user(features).items():
        genuine = frame.loc[get_labels(frame) == 1].sort_index(level="Test ID")
        if len(genuine) <= n_enroll:
            raise ValueError(f"User {user} needs genuine attempts for both enrollment and probing")
        enrollments.append(genuine.iloc[:n_enroll])
        probes.extend([genuine.iloc[n_enroll:], frame.loc[get_labels(frame) == 0]])
    return pd.concat(enrollments).sort_index(), pd.concat(probes).sort_index()


def build_comparison_vectors(features, n_enroll=5, template_strategy="median", absolute=True):
    enrollments, probes = split_enroll_probe(features, n_enroll)
    templates = enrollments.groupby(level="User ID").agg(template_strategy)
    repeated = templates.reindex(probes.index.get_level_values("User ID")).set_axis(probes.index)
    differences = probes - repeated
    return differences.abs() if absolute else differences


def _score(estimator, features):
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(features)
    elif hasattr(estimator, "score_samples"):
        scores = estimator.score_samples(features)
    elif hasattr(estimator, "predict_proba"):
        positive = np.flatnonzero(np.asarray(estimator.classes_) == 1)
        if len(positive) != 1:
            raise ValueError("A classifier must have a genuine (1) class")
        scores = estimator.predict_proba(features)[:, positive[0]]
    else:
        raise TypeError("Estimator needs decision_function, score_samples, or predict_proba")
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(features),) or not np.isfinite(scores).all():
        raise ValueError("Estimator must return one finite, higher-is-genuine score per probe")
    return scores


def _finish_scores(scores, folds=None, **metadata):
    scores = scores.dropna().sort_index().rename("score")
    scores.attrs.update(metadata)
    if folds is not None:
        scores.attrs["folds"] = {index: int(folds.loc[index]) for index in scores.index}
    return scores


def cross_val_predict_per_user(model, features, cv):
    """P1 scores in original attempt identity; each row must be tested once."""
    _validate_features(features)
    scores = pd.Series(np.nan, index=features.index)
    folds = pd.Series(-1, index=features.index, dtype=int)
    for _, frame in split_by_user(features).items():
        labels = get_labels(frame)
        for fold, (train_idx, test_idx) in enumerate(cv.split(frame, labels)):
            index = frame.index[test_idx]
            if (folds.loc[index] != -1).any() or np.intersect1d(train_idx, test_idx).size:
                raise ValueError("CV must test each attempt once with disjoint training rows")
            fitted = clone(model).fit(frame.iloc[train_idx], labels[train_idx])
            scores.loc[index] = _score(fitted, frame.iloc[test_idx])
            folds.loc[index] = fold
    if scores.isna().any():
        raise ValueError("CV did not score every attempt")
    return _finish_scores(scores, folds, protocol="p1")


def enroll_and_score(detector, features, n_enroll=5):
    enrollments, probes = split_enroll_probe(features, n_enroll)
    scores = pd.Series(index=probes.index, dtype=float)
    for user, frame in split_by_user(probes).items():
        fitted = clone(detector).fit(enrollments.loc[[user]])
        scores.loc[frame.index] = _score(fitted, frame)
    return _finish_scores(scores, protocol="p2", n_enroll=n_enroll)


def cross_val_predict_global(model, features, cv, n_enroll=5, typist_candidates=None):
    """P2 shared verifier with conservative identity separation.

    Candidates are tuples of every plausible typist, indexed by attempt. A row
    is usable only when its owner and ALL candidates are in the same fold side.
    Guarantees are conditional on the attribution window containing the typist.
    Missing identities are excluded, never assumed equal to the card owner.
    """
    if typist_candidates is None:
        raise ValueError("Global verification requires typist_candidates, not just card-owner groups")
    comparison = build_comparison_vectors(features, n_enroll=n_enroll)
    candidates = typist_candidates.reindex(comparison.index)
    if candidates.isna().any():
        raise ValueError("Typist candidates must cover every probe (use () for unknown)")
    owners = comparison.index.get_level_values("User ID")
    scores = pd.Series(np.nan, index=comparison.index)
    folds = pd.Series(-1, index=comparison.index, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(cv.split(comparison, get_labels(comparison), owners)):
        train_people, test_people = set(owners[train_idx]), set(owners[test_idx])
        if train_people & test_people:
            raise ValueError("Global CV must keep claimed users disjoint")
        train_idx = np.asarray([i for i in train_idx if candidates.iloc[i] and set(candidates.iloc[i]) <= train_people])
        test_idx = np.asarray([i for i in test_idx if candidates.iloc[i] and set(candidates.iloc[i]) <= test_people])
        if not len(train_idx) or not len(test_idx):
            raise ValueError("No identity-disjoint rows remain in this fold")
        if len(np.unique(get_labels(comparison.iloc[train_idx]))) != 2:
            raise ValueError("A global training fold needs genuine and impostor attempts")
        index = comparison.index[test_idx]
        if (folds.loc[index] != -1).any():
            raise ValueError("CV must test each attempt at most once")
        fitted = clone(model).fit(comparison.iloc[train_idx], get_labels(comparison.iloc[train_idx]))
        scores.loc[index] = _score(fitted, comparison.iloc[test_idx])
        folds.loc[index] = fold
    return _finish_scores(scores, folds, protocol="p2", n_enroll=n_enroll,
                          n_candidate_probes=len(comparison),
                          n_excluded_probes=int(scores.isna().sum()),
                          identity_policy="all plausible typists within fold")


class TemplateDetector(OutlierMixin, BaseEstimator):
    """Negative distance from an enrollment template, with enrollment-only scale.

    median/std matches z-scored Manhattan-to-median. mean/mad uses the mean
    absolute deviation from the mean. Zero spread uses scale 1; all-missing
    enrollment columns are imputed to zero.
    """

    def __init__(self, center="median", scale="std", metric="manhattan"):
        self.center = center
        self.scale = scale
        self.metric = metric

    def fit(self, X, y=None):
        if self.center not in {"mean", "median"} or self.scale not in {"std", "mad", None}:
            raise ValueError("Use center=mean/median and scale=std/mad/None")
        values = validate_data(self, X, ensure_all_finite="allow-nan")
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True).fit(values)
        values = self.imputer_.transform(values)
        self.center_ = (np.mean if self.center == "mean" else np.median)(values, axis=0)
        if self.scale == "std":
            spread = values.std(axis=0)
        elif self.scale == "mad":
            spread = np.abs(values - self.center_).mean(axis=0)
        else:
            spread = np.ones(values.shape[1])
        self.scale_ = np.where(spread > 0, spread, 1.)
        return self

    def score_samples(self, X):
        check_is_fitted(self, ["center_", "scale_", "imputer_"])
        values = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        values = self.imputer_.transform(values)
        return -pairwise_distances(values / self.scale_, (self.center_ / self.scale_)[None], metric=self.metric).ravel()

    def decision_function(self, X):
        return self.score_samples(X)
