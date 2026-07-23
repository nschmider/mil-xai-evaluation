import torch
from torch.utils.data import DataLoader
from torchmil.data import collate_fn


def single_perturbation_one_bag(X, model, device):
    """Helper function. Computes single perturbation scores for one bag.

    Args:
        X: The current bag
        model: The trained model
        device: Device used for computation

    Returns:
        Single perturbation scores for one bag
    """
    preds = []
    for patch_num in range(X.shape[1]):
        X_subset = X[:, patch_num : patch_num + 1].to(device)
        pred = model(X_subset).cpu()
        preds.append(pred)
    preds = torch.cat(preds).squeeze()  # Reshapes to a tensor of size (num_patches,)
    return preds


def single_perturbation(dataset, model, device):
    """Computes single perturbation by using the output of a single patch as its attribution score

    Args:
        dataset: The dataset
        model: The trained model
        device: Device used for computation

    Returns:
        Single perturbation scores
    """
    scores = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for bag in dataloader:
        X = bag["X"]
        preds = single_perturbation_one_bag(X, model, device)
        scores.append(preds)
    return scores
