import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import torch

from src.utils import min_max_normalization


def compute_numerical_metrics(scores, patch_labels):
    """Computes numerical metrics like AUROC and Average precision

    Args:
        scores: Attribution scores
        patch_labels: Patch labels

    Returns:
        Metrics dict containing the metrics
    """
    slide_labels = torch.tensor(
        [slide_patch_labels.sum() > 0 for slide_patch_labels in patch_labels]
    )
    positive_scores = torch.cat(
        tuple(
            [
                score.cpu().flatten()
                for score, slide_label in zip(scores, slide_labels)
                if slide_label == 1
            ]
        )
    )
    negative_scores = torch.cat(
        tuple(
            [
                score.cpu().flatten()
                for score, slide_label in zip(scores, slide_labels)
                if slide_label == 0
            ]
        )
    )
    positive_labels = torch.cat(
        tuple(
            [
                slide_patch_label.cpu().flatten()
                for slide_patch_label, slide_label in zip(patch_labels, slide_labels)
                if slide_label == 1
            ]
        )
    )
    negative_labels = torch.cat(
        tuple(
            [
                slide_patch_label.cpu().flatten()
                for slide_patch_label, slide_label in zip(patch_labels, slide_labels)
                if slide_label == 0
            ]
        )
    )
    scores = torch.cat(tuple([score.cpu().flatten() for score in scores]))
    patch_labels = torch.cat(
        tuple([patch_label.cpu().flatten() for patch_label in patch_labels])
    )
    positive_rate = patch_labels.mean().item()
    positive_labels_flattened = torch.cat(
        tuple([patch_label.cpu().flatten() for patch_label in positive_labels])
    )
    tumor_positive_rate = positive_labels_flattened.float().mean().item()

    tumor_scores = scores[patch_labels == 1]
    normal_scores = scores[patch_labels == 0]

    # AUROC score
    auroc = roc_auc_score(patch_labels, scores)

    # Average Precision score
    ap = average_precision_score(patch_labels, scores)

    ### Normalized ###
    norm_scores = min_max_normalization(scores)
    # AUROC score
    norm_auroc = roc_auc_score(patch_labels, norm_scores)

    # Average Precision score
    norm_ap = average_precision_score(patch_labels, norm_scores)

    # Pearson correlation
    combined = torch.stack((norm_scores, patch_labels))
    corr = torch.corrcoef(combined)
    norm_corr_value = corr[0, 1].item()

    norm_fpr, norm_tpr, norm_roc_thresholds = roc_curve(patch_labels, norm_scores)
    norm_precision, norm_recall, norm_pr_thresholds = precision_recall_curve(
        patch_labels, norm_scores
    )

    ### Test with positive and negative instances ###

    # Positive
    auroc_pos = roc_auc_score(positive_labels, positive_scores)
    ap_pos = average_precision_score(positive_labels, positive_scores)

    fpr_pos, tpr_pos, roc_thresholds_pos = roc_curve(positive_labels, positive_scores)
    precision_pos, recall_pos, pr_thresholds_pos = precision_recall_curve(
        positive_labels, positive_scores
    )

    # Negative
    auroc_neg = roc_auc_score(negative_labels, negative_scores)
    ap_neg = average_precision_score(negative_labels, negative_scores)

    # Plot
    fpr, tpr, roc_thresholds = roc_curve(patch_labels, scores)

    precision, recall, pr_thresholds = precision_recall_curve(patch_labels, scores)

    return {
        "auroc": auroc,
        "average_precision": ap,
        "positive_labels": positive_labels,
        "normalized_auroc": norm_auroc,
        "normalized_average_precision": norm_ap,
        "normalized_correlation": norm_corr_value,
        "positive_patch_ratio": positive_rate,
        "tumor_patch_ratio": tumor_positive_rate,
        "tumor_score_mean": tumor_scores.mean().item(),
        "normal_score_mean": normal_scores.mean().item(),
        "tumor_score_median": tumor_scores.median().item(),
        "normal_score_median": tumor_scores.median().item(),
        "positive_slide_auroc": auroc_pos,
        "positive_slide_ap": ap_pos,
        "negative_slide_auroc": auroc_neg,
        "negative_slide_ap": ap_neg,
        "roc": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        },
        "pr": {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
        },
        "roc_pos": {
            "fpr": fpr_pos,
            "tpr": tpr_pos,
            "thresholds": roc_thresholds_pos,
        },
        "pr_pos": {
            "precision": precision_pos,
            "recall": recall_pos,
            "thresholds": pr_thresholds_pos,
        },
        "norm_roc": {
            "fpr": norm_fpr,
            "tpr": norm_tpr,
            "thresholds": norm_roc_thresholds,
        },
        "norm_pr": {
            "precision": norm_precision,
            "recall": norm_recall,
            "thresholds": norm_pr_thresholds,
        },
        "scores": {
            "tumor": tumor_scores,
            "normal": normal_scores,
            "all": scores,
        },
    }


def print_numerical_metrics(metrics):
    """Prints information to numerical metrics

    Args:
        metrics: Metrics dict from function compute_numerical_metrics()
    """
    print("Positive patch label ratio:", metrics["positive_patch_ratio"])
    print("Positive patch label ratio in tumor slides:", metrics["tumor_patch_ratio"])

    print("Mean attribution of positive patches:", metrics["tumor_score_mean"])
    print("Mean attribution of negative patches:", metrics["normal_score_mean"])
    print("Median attribution of positive patches:", metrics["tumor_score_median"])
    print("Median attribution of negative patches:", metrics["normal_score_median"])

    print("_" * 30)
    print("Normalized scores")
    print("_" * 30)
    print("AUROC with normalized scores:", metrics["normalized_auroc"])
    print(
        "Average precision with normalized scores:",
        metrics["normalized_average_precision"],
    )
    print(
        "Pearson correlation with normalized scores:", metrics["normalized_correlation"]
    )

    print("_" * 30)
    print("Test with positive and negative slides")
    print("_" * 30)

    print("AUROC for positive slides:", metrics["positive_slide_auroc"])
    print("Average Precision for positive slides:", metrics["positive_slide_ap"])

    print("AUROC for negative slides:", metrics["negative_slide_auroc"])
    print("Average Precision for negative slides:", metrics["negative_slide_ap"])

    print("_" * 30)
    print("Additional information")
    print("_" * 30)
    print("Max Scores:", metrics["scores"]["all"].max())
    print("Min Scores:", metrics["scores"]["all"].min())
    print("len(thresholds)", len(metrics["roc"]["thresholds"]))
    print("Unique values:", len(torch.unique(metrics["scores"]["all"])))
    print("Thresholds:", metrics["roc"]["thresholds"])
