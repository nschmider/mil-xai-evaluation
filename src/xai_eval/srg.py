from matplotlib import pyplot as plt
import numpy as np
import torch
from torchmil.datasets import CAMELYON16MILDataset
import warnings

from src.utils import evaluate_mask

warnings.filterwarnings("ignore")


def srg(scores, model, device, num_bins=100, plot=True):
    """
    Symmetric Relevance Gain, metric to compare the change in prediction when removing most important features first or last.
    Uses bins that are progressively removed.

    Args:
        scores: Attribution scores
        model: The trained model
        device: CUDA or CPU
        num_bins: Number of bins. Defaults to 100.
        plot: Whether to plot the model prediction curves. Defaults to True.

    Returns:
        The Symmetric Relevance Gain
    """
    asc, asc_preds = aupc(
        scores, model, descending=False, device=device, num_bins=num_bins
    )
    desc, desc_preds = aupc(
        scores, model, descending=True, device=device, num_bins=num_bins
    )
    if plot:
        X = range(num_bins + 1)
        plt.figure()
        plt.plot(
            X, np.mean(np.array(asc_preds), axis=-1).reshape(-1), label="Ascending"
        )
        plt.plot(
            X, np.mean(np.array(desc_preds), axis=-1).reshape(-1), label="Descending"
        )

        plt.xlabel("Dropped fraction")
        plt.ylabel("Model prediction")
        plt.legend()
        plt.savefig("results/plots/srg/curve_plots.png")
        plt.close()
    return {"srg": asc - desc, "ascending": asc, "descending": desc}


def aupc(scores, model, descending, device, num_bins=100):
    """
    Generates prediction for progressively removed patches in descending or ascending order according to attribution.

    Args:
        scores: The attribution scores
        model: The trained model
        descending: The order in which to remove the patches
        device: CUDA or CPU
        num_bins: Number of bins. Defaults to 100.

    Returns:
        Area under the Perturbation curve (Average of predictions)
    """
    dataset = CAMELYON16MILDataset(root="data", features="UNI", partition="test")
    binned_scores = []
    masks = [[] for _ in range(num_bins + 1)]
    masks[num_bins] = [[] for _ in range(len(scores))]

    # Bin scores
    for score in scores:  # iterates over bags
        binned_index = torch.argsort(score, dim=0, descending=descending)
        binned_scores = torch.tensor_split(
            binned_index, num_bins
        )  # split into num_bins bins, shape (num_bins, num_patches)
        for bin in range(num_bins):
            kept = torch.cat(binned_scores[bin:])
            masks[bin].append(kept)

    output, preds = evaluate_mask(dataset, masks, model, device)

    return output, preds
