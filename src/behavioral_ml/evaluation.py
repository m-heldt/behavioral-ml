"""Verification metrics for scores where larger values mean more genuine."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import DetCurveDisplay, roc_auc_score, roc_curve


def _eer_from_curve(far, frr, thresholds):
    """Use the full ROC point nearest FAR=FRR; average its two error rates.

    This is a discrete estimate, not an interpolated operating threshold.
    """
    idx = np.argmin(np.abs(np.asarray(far) - np.asarray(frr)))
    return float((far[idx] + frr[idx]) / 2), float(thresholds[idx])


def _roc(y_true, y_score):
    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and scores must be aligned one-dimensional arrays")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Verification metrics require both genuine and impostor labels")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")
    return roc_curve(labels, scores, drop_intermediate=False)


def eer_score(y_true, y_score):
    far, tar, thresholds = _roc(y_true, y_score)
    return _eer_from_curve(far, 1 - tar, thresholds)[0]


def threshold_at_far(y_true, y_score, far=0.01):
    """Select the best observed recall subject to FAR <= target.

    Use development/calibration scores here for a deployable threshold.
    A threshold chosen on evaluation scores is descriptive only.
    """
    if not 0 <= far <= 1:
        raise ValueError("far must be between 0 and 1")
    fpr, tpr, thresholds = _roc(y_true, y_score)
    eligible = np.flatnonzero(fpr <= far)
    idx = eligible[np.argmax(tpr[eligible])]
    return float(thresholds[idx])


def metrics_at_threshold(y_true, y_score, threshold):
    """Evaluate a previously selected threshold without selecting it again."""
    _roc(y_true, y_score)
    labels = np.asarray(y_true, dtype=bool)
    accepted = np.asarray(y_score) >= threshold
    return {"far": float(accepted[~labels].mean()),
            "frr": float((~accepted[labels]).mean())}


def frr_at_far(y_true, y_score, far=0.01):
    """Descriptive FRR at an attainable ROC point with FAR <= target."""
    threshold = threshold_at_far(y_true, y_score, far)
    return metrics_at_threshold(y_true, y_score, threshold)["frr"]


def verification_report(y_true, y_score):
    far, tar, thresholds = _roc(y_true, y_score)
    frr = 1 - tar
    eer, eer_threshold = _eer_from_curve(far, frr, thresholds)
    return {"far": far, "tar": tar, "frr": frr, "thresholds": thresholds,
            "auc": float(roc_auc_score(y_true, y_score)), "eer": eer,
            "eer_threshold": eer_threshold}


def _validate_scores(scores):
    if not isinstance(scores, pd.Series) or not isinstance(scores.index, pd.MultiIndex):
        raise ValueError("Expected a score Series indexed by user, label, and attempt")
    if list(scores.index.names) != ["User ID", "Is Genuine", "Test ID"]:
        raise ValueError("Score index must be (User ID, Is Genuine, Test ID)")
    if scores.empty or not scores.index.is_unique or not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise ValueError("Score indices must be nonempty/unique and values finite")


def per_user_metrics(scores, metrics=None):
    """One row per user, including class counts; one-class metrics are NaN."""
    _validate_scores(scores)
    if metrics is None:
        metrics = {"auc": roc_auc_score, "eer": eer_score}
    rows = {}
    for user, group in scores.groupby(level="User ID", sort=True):
        labels = group.index.get_level_values("Is Genuine").to_numpy(dtype=int)
        row = {"n_genuine": int(labels.sum()), "n_impostor": int((labels == 0).sum())}
        for name, metric in metrics.items():
            row[name] = float(metric(labels, group)) if len(np.unique(labels)) == 2 else np.nan
        rows[user] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("User ID")


def summarize(scores, far=0.01):
    """Keep pooled and per-user statistics explicitly separate."""
    _validate_scores(scores)
    labels = scores.index.get_level_values("Is Genuine").to_numpy(dtype=int)
    per_user = per_user_metrics(scores)
    report = verification_report(labels, scores)
    return pd.Series({
        "n_probes": len(scores), "n_genuine": int(labels.sum()),
        "n_impostor": int((labels == 0).sum()),
        "n_users_with_both_classes": int(per_user["eer"].notna().sum()),
        "pooled_auc": report["auc"], "pooled_eer": report["eer"],
        "per_user_auc_mean": per_user["auc"].mean(),
        "per_user_auc_median": per_user["auc"].median(),
        f"pooled_frr_at_far_{far:g}": frr_at_far(labels, scores, far),
        "per_user_eer_mean": per_user["eer"].mean(),
        "per_user_eer_median": per_user["eer"].median(),
        "per_user_eer_iqr": per_user["eer"].quantile(.75) - per_user["eer"].quantile(.25),
    })


def plot_roc_curve(report, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(report["far"], report["tar"], label=label or f"AUC = {report['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False Acceptance Rate (FAR)", ylabel="True Acceptance Rate (TAR)", title="ROC curve")
    ax.legend()
    ax.grid(alpha=.3)
    return ax


def plot_far_frr_vs_threshold(report, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    finite = np.isfinite(report["thresholds"])
    ax.plot(report["thresholds"][finite], report["far"][finite], label="FAR")
    ax.plot(report["thresholds"][finite], report["frr"][finite], label="FRR")
    ax.axvline(report["eer_threshold"], linestyle="--", label="Descriptive EER threshold")
    ax.set(xlabel="Decision threshold", ylabel="Error rate", title="Error vs threshold")
    ax.legend()
    return ax


def plot_det_curve(report, ax=None, label=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    DetCurveDisplay(fpr=report["far"], fnr=report["frr"], estimator_name=label).plot(ax=ax)
    ax.set(xlabel="False Acceptance Rate (FAR)", ylabel="False Rejection Rate (FRR)", title="DET curve")
    return ax


def plot_det_curves(experiments, ax=None):
    for name, scores in experiments.items():
        labels = scores.index.get_level_values("Is Genuine")
        ax = plot_det_curve(verification_report(labels, scores), ax=ax, label=name)
    return ax


def plot_per_user_eer(per_user, n_worst=5, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    values = per_user["eer"].dropna()
    ax.hist(values, bins=np.linspace(0, 1, 21))
    worst = values.nlargest(n_worst)
    annotation = "Highest EERs\n" + "\n".join(f"User {u}: {v:.3f}" for u, v in worst.items())
    ax.text(.98, .98, annotation, transform=ax.transAxes, ha="right", va="top")
    ax.set(xlabel="Per-user EER (discrete estimate)", ylabel="Users", title="Variation across users")
    return ax


# Compatibility for existing exploration cells; new code uses the names above.
get_eer = _eer_from_curve
get_eer_from_scores = eer_score
get_metrics = verification_report
plot_eer_threshold = plot_far_frr_vs_threshold
