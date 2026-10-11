import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, DetCurveDisplay

def get_eer(far, frr, thresholds):
    idx = np.argmin(np.abs(far - frr))

    eer = (far[idx] + frr[idx]) / 2
    eer_threshold = thresholds[idx]

    print(f"EER: {eer:.3f}")
    print(f"EER threshold: {eer_threshold:.3f}")

    return eer, eer_threshold


def get_metrics(y_true, y_score):
    far, tar, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    frr = 1 - tar

    eer, eer_threshold = get_eer(far, frr, thresholds)

    metrics = {}
    metrics["far"] = far
    metrics["tar"] = tar
    metrics["thresholds"] = thresholds
    metrics["auc"] = auc
    metrics["frr"] = frr
    metrics["eer"] = eer
    metrics["eer_threshold"] = eer_threshold
    
    return metrics



def plot_roc_curve(metrics):
    plt.figure(figsize=(7, 5))
    plt.plot(metrics["far"], metrics["tar"], label=f"AUC = {metrics["auc"]:.3f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Acceptance Rate (FAR)")
    plt.ylabel("True Acceptance Rate (TAR)")
    plt.title("ROC curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_eer_threshold(metrics):
    plt.figure(figsize=(7, 5))

    plt.plot(metrics["thresholds"], metrics["far"], label="FAR")
    plt.plot(metrics["thresholds"], metrics["frr"], label="FRR")
    plt.axvline(metrics["eer_threshold"], linestyle="--", label=f"EER threshold = {metrics["eer_threshold"]:.2f}")

    plt.xlabel("Decision threshold")
    plt.ylabel("Error rate")
    plt.title("Authentication error vs Decision threshold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_det_curve(metrics, classifier_name):
    _, ax = plt.subplots(figsize=(7, 5))
    DetCurveDisplay(
        fpr=metrics["far"], 
        fnr=metrics["frr"], 
        classifier_name=classifier_name
        ).plot(ax=ax)

    ax.set_xlabel("False Acceptance Rate (FAR)")
    ax.set_ylabel("False Rejection Rate (FRR)")
    plt.title("DET curve")
    plt.show()
