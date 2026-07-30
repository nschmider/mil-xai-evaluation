from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmil.data import collate_fn
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def min_max_normalization(scores):
    denom = scores.max() - scores.min()
    if denom == 0:
        normalized_scores = torch.ones_like(scores) / len(scores)
    else:
        normalized_scores = (scores - scores.min()) / denom  # Min-max normalization
    return normalized_scores


def evaluate_mask(dataset, masks, model, device):
    """
    Makes prediction (and averages them) for a mask of order (# masks, # bags, # patches in bag).
    Mask contains the patches to keep for the prediction.

    Args:
        dataset: The dataset to make predictions on
        masks: The mask to evaluate
        model: The trained model
        device: CUDA or CPU

    Returns:
        The averaged predictions
    """
    preds = [[] for _ in range(len(masks))]
    with torch.no_grad():
        for mask_num, bag_mask in enumerate(tqdm(masks)):
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
            )
            for i, batch in enumerate(dataloader):
                X = batch["X"].to(device)
                # inverse_mask = [patch for patch in range(X.shape[1]) if patch not in bag_mask[i]] # patches to be masked
                # mask[:, inverse_mask] = False
                if bag_mask[i] == []:
                    X = torch.zeros_like(X)
                else:
                    X = X[:, bag_mask[i]]
                Y_pred = model(X)
                Y_pred = torch.sigmoid(Y_pred)
                preds[mask_num].append(Y_pred.item())
    return np.mean(np.hstack(preds)), preds


def shape_test(dataset, scores):
    for method, method_scores in scores.items():
        assert len(method_scores) == len(dataset)
        all_good = True

        for i, bag in enumerate(dataset):
            num_patches = bag["X"].shape[0]

            if len(method_scores[i]) != num_patches:
                print(
                    f"{method}: Bag {i} has {num_patches} patches "
                    f"but {len(method_scores[i])} scores."
                )
                all_good = False

        if all_good:
            print("All perturbation scores have the correct shape.")
