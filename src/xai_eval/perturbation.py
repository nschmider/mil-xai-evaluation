import math

from sklearn.linear_model import LinearRegression, Ridge
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
        X = X.to(device)
        if method == "one-removed":
            pred_total = model(X).detach().cpu()
        for patch_num in range(X.shape[1]):
            if method == "single":  # Computes the prediction for a single patch
                X_subset = X[:, patch_num : patch_num + 1]
                pred = model(X_subset).detach().cpu()
            elif method == "one-removed":  # Leaves out the current bag for a prediction
                indices = torch.arange(X.shape[1])
                indices = indices[indices != patch_num]
                X_subset = X[:, indices]
                pred_subset = model(X_subset).detach().cpu()
                pred = pred_total - pred_subset
            else:
                raise ValueError(f"Unknown perturbation method: {method}")
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
    return {"combined": scores, "single": single, "one_removed": one_removed}


def milli(dataset, model, n_masks, initial_scores, device, alpha, beta):
    """Performs MILLI perturbation with hyperparameters alpha and beta.
    Samples coalitions of patches and fits a linear classifier to predict from the cohort to a prediction.
    Returns the classifier's coefficients as attribution scores.

    Args:
        dataset: The dataset
        model: The trained model
        n_masks: Number of cohorts to sample per bag
        initial_scores: Perturbation scores as heuristic for sampling
        device: Device used for computation
        alpha: Hyperparameter alpha, if alpha < 0.5, lower relevance instances are preferred in sampling
        beta: Hyperparameter beta, higher beta prefers bigger coalitions

    Returns:
        The classifier's coefficients
    """
    model.eval()
    pi_rs = []
    scores = []

    beta_hat = beta if alpha < 0.5 else -beta
    for score in initial_scores:
        r = compute_initial_ranking(score)
        k = len(r)
        if beta_hat >= 0:
            pi_r = (2 * alpha - 1) * (1 - r / k) * torch.exp(-beta_hat * r) + 1 - alpha
        else:
            pi_r = (1 - 2 * alpha) * (1 + (r - k) / k) * torch.exp(
                torch.abs(torch.tensor(beta_hat)) * (r - k)
            ) + alpha
        pi_rs.append(pi_r)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for bag_num, bag in enumerate(tqdm(dataloader)):
            masks = []
            preds = []
            pi_r = pi_rs[bag_num]
            pi_m = []
            for _ in range(n_masks):
                X = bag["X"].to(device)

                # Sample patches
                mask = torch.bernoulli(pi_r)
                while mask.sum() == 0:
                    mask = torch.bernoulli(pi_r)
                weight = 1 / mask.sum() * torch.sum(mask * pi_r)
                X_subset = X[:, mask.bool()]

                # Predict logits
                pred = model(X_subset)

                masks.append(mask.cpu())
                preds.append(pred.cpu())
                pi_m.append(weight.item())

            # Fit model
            lr = LinearRegression()
            masks = torch.stack(masks).numpy()
            preds = torch.stack(preds).numpy()
            lr.fit(masks, preds, sample_weight=pi_m)

            # Extract scores
            coefs = torch.tensor(lr.coef_).squeeze()
            scores.append(coefs)
    return scores


def rise(dataset, model, n_masks, p, device):
    """RISE. Averages the predictions of all coalitions containing a certain patch

    Args:
        dataset: The dataset
        model: The trained model
        n_masks: Number of masks to be computed
        p: Probability for a patch to be sampled
        device: Device used for computation

    Returns:
        RISE attribution score
    """
    scores = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for bag in tqdm(dataloader):
            masks = []
            preds = []
            for _ in range(n_masks):
                X = bag["X"].to(device)

                # Sample patches
                mask = torch.bernoulli(torch.full((X.shape[1],), p))
                while mask.sum() == 0:
                    mask = torch.bernoulli(torch.full((X.shape[1],), p))
                X_subset = X[:, mask.bool()]

                # Predict logits
                pred = model(X_subset)

                masks.append(mask.unsqueeze(0))
                preds.append(pred.unsqueeze(0))

            masks = torch.cat(masks)  # (masks, patches)
            masks = masks.to(device)
            preds = torch.cat(preds)  # (masks, 1)

            # Compute the predictions where the mask is active
            masked_preds = masks * preds  # (masks, patches)
            # Aggregate over masks
            expected_score_sum = torch.sum(masked_preds, dim=0)  # (patches,)
            expected_score = expected_score_sum / (torch.sum(masks, dim=0) + 1e-8)
            scores.append(expected_score.cpu())
    return scores


def lime(dataset, model, n_masks, device):
    """Performs LIME.
    Samples coalitions of patches and fits a linear classifier to predict from the cohort to a prediction.
    Returns the classifier's coefficients as attribution scores.

    Args:
        dataset: The dataset
        model: The trained model
        n_masks: Number of cohorts to sample per bag
        device: Device used for computation
        
    Returns:
        The classifier's coefficients
    """
    model.eval()
    pi_z = []
    scores = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for _, bag in enumerate(tqdm(dataloader)):
            masks = []
            preds = []
            pi_z = []

            X = bag["X"].to(device)
            num_patches = X.shape[1]
            masks.append(torch.ones(num_patches))
            preds.append(model(X).cpu())
            pi_z.append(1.0)

            for _ in range(n_masks):
                # Sample patches
                num_patches_to_sample = torch.randint(
                    low=1, high=num_patches + 1, size=(1,)
                ).item()
                mask = torch.randperm(num_patches)[:num_patches_to_sample]
                bin_mask = torch.zeros((num_patches,))
                bin_mask[mask] = 1

                distance = num_patches - num_patches_to_sample
                sigma = 0.25 * math.sqrt(num_patches)
                weight = torch.exp(torch.tensor(-distance / sigma**2))

                X_subset = X[:, mask]

                # Predict logits
                pred = model(X_subset)

                masks.append(bin_mask.cpu())
                preds.append(pred.cpu())
                pi_z.append(weight.item())

            # Fit model
            lr = Ridge()
            masks = torch.stack(masks).numpy()
            preds = torch.stack(preds).numpy().squeeze()
            lr.fit(masks, preds, sample_weight=pi_z)

            # Extract scores
            coefs = torch.tensor(lr.coef_).squeeze()
            scores.append(coefs)
    return scores


def compute_initial_ranking(scores):
    """Computes the rank of the score.
    Example: compute_initial_ranking([1, 4, 2, -1, 6, 0]) -> [3, 1, 2, 5, 0, 4]

    Args:
        scores: Initial scores

    Returns:
        Initial rank of the scores
    """
    ranking = torch.argsort(scores, descending=True)
    r = torch.zeros_like(ranking)
    r[ranking] = torch.arange(len(ranking))
    return r
