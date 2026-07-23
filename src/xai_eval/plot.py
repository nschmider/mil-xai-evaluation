import matplotlib.pyplot as plt


def plot_attention_dist(metrics, path):
    """Generates histogram of attention distribution of positive/negative patches respectively

    Args:
        metrics : Metrics computed by function numerical_metrics.compute_numerical_metrics()
        path : Path to save the plot
    """
    normal_scores = metrics["scores"]["normal"]
    tumor_scores = metrics["scores"]["tumor"]

    plt.hist(
        normal_scores.numpy(),
        bins=100,
        alpha=0.5,
        label="Non-tumor patches",
        density=True,
    )

    plt.hist(
        tumor_scores.numpy(),
        bins=100,
        alpha=0.5,
        label="Tumor patches",
        density=True,
    )

    plt.xlabel("Attention score")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(path)
    plt.close()


def plot_roc(metrics, path):
    """Generates ROC comparing the attribution scores and patch labels

    Args:
        metrics : Metrics computed by function numerical_metrics.compute_numerical_metrics()
        path : Path to save the plot
    """
    auroc = metrics["auroc"]
    fpr = metrics["roc"]["fpr"]
    tpr = metrics["roc"]["tpr"]

    plt.figure()
    plt.step(fpr, tpr, where="post", alpha=0.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC, AUROC={auroc:0.2f}")
    plt.savefig(path)
    plt.close()


def plot_prc(metrics, path):
    """Generates Precision-Recall curve comparing the attribution scores and patch labels

    Args:
        metrics : Metrics computed by function numerical_metrics.compute_numerical_metrics()
        path : Path to save the plot
    """
    ap = metrics["average_precision"]
    recall = metrics["pr"]["recall"]
    precision = metrics["pr"]["precision"]
    plt.figure()
    plt.step(recall, precision, where="post", alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve, Average Precision={ap:0.2f}")
    plt.savefig(path)
    plt.close()


def plot_normalized_roc(metrics, path):
    """Generates ROC comparing the normalized attribution scores and patch labels

    Args:
        metrics : Metrics computed by function numerical_metrics.compute_numerical_metrics()
        path : Path to save the plot
    """
    auroc = metrics["normalized_auroc"]
    fpr = metrics["norm_roc"]["fpr"]
    tpr = metrics["norm_roc"]["tpr"]
    plt.figure()
    plt.step(fpr, tpr, where="post", alpha=0.2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC, AUROC={auroc:0.2f}")
    plt.savefig(path)
    plt.close


def plot_normalized_prc(metrics, path):
    """Generates Precision-Recall curve comparing the normalized attribution scores and patch labels

    Args:
        metrics : Metrics computed by function numerical_metrics.compute_numerical_metrics()
        path : Path to save the plot
    """
    ap = metrics["normalized_average_precision"]
    recall = metrics["norm_pr"]["recall"]
    precision = metrics["norm_pr"]["precision"]
    plt.figure()
    plt.step(recall, precision, where="post", alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curve, Average Precision={ap:0.2f}")
    plt.savefig(path)
    plt.close()


def plot_all_numerical_metrics(metrics, output_dir):
    plot_attention_dist(metrics, f"{output_dir}/attention_distribution.png")

    plot_roc(metrics, f"{output_dir}/roc.png")

    plot_prc(metrics, f"{output_dir}/prc.png")

    plot_normalized_roc(metrics, f"{output_dir}/roc_normalized.png")

    plot_normalized_prc(metrics, f"{output_dir}/prc_normalized.png")
