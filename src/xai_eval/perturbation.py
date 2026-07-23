import torch
from torch.utils.data import DataLoader
from torchmil.data import collate_fn
from tqdm import tqdm


def perturbation_one_bag(X, model, device, method):
    """Helper function. Computes perturbation scores for one bag.

    Args:
        X: The current bag
        model: The trained model
        device: Device used for computation
        method: Perturbation method. Choice between "single" and "one-removed"

    Returns:
        Perturbation scores for one bag
    """
    preds = []
    model.eval()
    with torch.no_grad():
        if method == "one-removed":
            pred_total = torch.sigmoid(model(X.to(device))).detach().cpu()
        for patch_num in range(X.shape[1]):
            if method == "single":  # Computes the prediction for a single patch
                X_subset = X[:, patch_num : patch_num + 1].to(device)
                pred = model(X_subset).detach().cpu()
            if method == "one-removed":  # Leaves out the current bag for a prediction
                X = X.to(device)
                indices = torch.arange(X.shape[1])
                indices = indices[indices != patch_num]
                X_subset = X[:, indices]
                pred_subset = torch.sigmoid(model(X_subset)).detach().cpu()
                pred = pred_total - pred_subset
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
    for bag in tqdm(dataloader, desc="Single perturbation"):
        X = bag["X"]
        preds = perturbation_one_bag(X, model, device, method="single")
        scores.append(preds)
    return scores


def one_removed_perturbation(dataset, model, device):
    """Computes one-removed perturbation by using the difference of the output of the full bag with the output of the bag with the current element removed as its attribution score

    Args:
        dataset: The dataset
        model: The trained model
        device: Device used for computation

    Returns:
        One-removed perturbation scores
    """
    scores = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for bag in tqdm(dataloader, desc="One-removed perturbation"):
        X = bag["X"]
        preds = perturbation_one_bag(X, model, device, method="one-removed")
        scores.append(preds)
    return scores


def combined_perturbation(dataset, model, device):
    """Computes combined perturbation by computing the mean between the single and one-removed perturbation scores

    Args:
        dataset: The dataset
        model: The trained model
        device: Device used for computation

    Returns:
        Combined perturbation scores
    """
    scores = []
    single = []
    one_removed = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for bag in tqdm(dataloader, desc="Combined perturbation"):
        X = bag["X"]
        preds_single = perturbation_one_bag(X, model, device, method="single")
        preds_one_removed = perturbation_one_bag(X, model, device, method="one-removed")
        preds_combined = 0.5 * (preds_single + preds_one_removed)
        scores.append(preds_combined)
        single.append(preds_single)
        one_removed.append(preds_one_removed)
    return {
        "combined": scores,
        "single": single,
        "one_removed": one_removed
    }
