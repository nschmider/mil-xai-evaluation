from pathlib import Path
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmil.data import collate_fn
from tqdm import tqdm

from src.utils import min_max_normalization


def importance_sampling_mask(scores, n_masks):
    positive_masks = []
    negative_masks = []

    # Compute weights
    for score in scores:
        num_samples = np.random.randint(1, len(score))
        weights = min_max_normalization(score)

    # Compute positive masks
    for _ in range(n_masks // 2):
        masks = []
        for score in scores:
            mask = torch.multinomial(
                weights, num_samples=num_samples, replacement=False
            )
            masks.append(mask)
        positive_masks.append(masks)

    # Compute negative masks
    for _ in range(n_masks // 2):
        masks = []
        for score in scores:
            mask = torch.multinomial(
                1 - weights, num_samples=num_samples, replacement=False
            )
            masks.append(mask)
        negative_masks.append(masks)

    return positive_masks, negative_masks


def importance_sampling_multiple_bags(
    dataset, scores, model, metric, n_masks, device, plot=False, method=None, step=1
):
    """Importance sampling for multiple bags.

    Args:
        dataset: The dataset
        scores: Attribution scores
        model: The trained model
        metric: Sample patches according to relevance or in reverse order
        n_masks: Number of masks to smooth prediction
        device: Device used for computation
        plot: Whether to plot the performance curve with respect to cardinality. Defaults to False.

    Returns:
        The averaged relevance score for the desired relevance ordering
    """
    method = "unknown" if not method else method
    path = Path(f"results/tmp/{method}/{metric}_checkpoint.pkl")
    preds = []
    iteration = -1
    if path.exists():
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
            iteration = checkpoint["iteration"]
            preds = checkpoint["preds"]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for i, (bag, score) in enumerate(zip(dataloader, scores)):
        if i <= iteration:
            continue
        X = bag["X"]
        pred = importance_sampling_one_bag(
            X,
            model,
            score,
            metric,
            n_masks,
            i,
            device,
            plot,
            method,
            step,
            bag["Y"].item(),
        )
        preds.append(pred)
        checkpoint = {"iteration": i, "preds": preds}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"results/tmp/{method}/{metric}_checkpoint.pkl", "wb") as f:
            pickle.dump(checkpoint, f)
    return sum(preds) / len(preds)


def importance_sampling_one_bag(
    X,
    model,
    scores,
    metric,
    n_masks,
    bag_num,
    device,
    plot=False,
    method=None,
    step=1,
    slide_label=-1,
):
    """Importance sampling for one bag.
    Samples n_masks times with cardinalities 1 to |N|.
    Uses attribution scores as weights, or 1 - attribution scores in the case of metric "R-MIF"

    Args:
        X: The current bag patches
        model: The trained model
        scores: Attribution scores
        metric: Sample patches according to relevance or in reverse order
        n_masks: Number of masks to smoothe prediction
        bag_num: Index of the current bag in the dataset
        device: Device used for computation
        plot: Whether to plot the performance curve with respect to cardinality. Defaults to False.
        slide_label: Whether the slide is positive or negative, for plotting. Defaults to -1.

    Returns:
        The averaged relevance score for the desired relevance ordering
    """
    method = "unknown" if not method else method
    preds = []

    # Compute weights
    weights = min_max_normalization(scores)
    if metric == "R-MIF":
        weights = 1 - weights

    for cardinality in tqdm(range(1, X.shape[1] + 1, step), desc=f"[Bag {bag_num}]"):
        subset_preds = []
        for _ in range(n_masks):
            # print("number of patches:", len(weights))
            # print("cardinality:", cardinality)
            # print("positive weights:", (weights > 0).sum().item())
            # print("min:", weights.min().item())
            # print("max:", weights.max().item())
            # print("len(scores):", len(scores))
            # print("weights.numel():", weights.numel())
            # print("X.shape:", X.shape)
            mask = torch.multinomial(
                weights, num_samples=cardinality, replacement=False
            )
            with torch.no_grad():
                pred = torch.sigmoid(model(X[:, mask].to(device))).cpu()
            subset_preds.append(pred)
        preds.append(sum(subset_preds) / len(subset_preds))

    if plot:
        x = range(len(preds))
        plt.figure()
        plt.plot(x, preds, label=f"{metric}, Bag {bag_num}")
        plt.xlabel("Cardinality")
        plt.ylabel("Average Prediction Likelihood")
        plt.legend()
        path = Path(
            f"results/plots/rsrg/{method}/importance_sampling_{metric}_{bag_num}_{slide_label}.png"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        plt.close()

    return sum(preds) / len(preds)


def rsrg(model, scores, n_masks, dataset, device, plot=False, method=None, step=1):
    """Computes Relative Symmetric Relevance Gain.
    Samples patches according to their attribution scores or inversely, computes the difference.

    Args:
        model: The trained model
        scores: Attribution scores
        n_masks: Number of masks to smoothe prediction
        dataset: The dataset
        device: Device used for computation
        plot: Whether to plot the performance curve with respect to cardinality. Defaults to False.

    Returns:
        Relative Symmetric Relevance Gain, averaged across slides
    """
    t1 = time.time()
    r_lif = importance_sampling_multiple_bags(
        dataset, scores, model, "R-LIF", n_masks, device, plot, method, step
    )
    r_mif = importance_sampling_multiple_bags(
        dataset, scores, model, "R-MIF", n_masks, device, plot, method, step
    )
    t2 = time.time()
    return {"RSRG": r_lif - r_mif, "R-LIF": r_lif, "R-MIF": r_mif, "time": t2 - t1}
