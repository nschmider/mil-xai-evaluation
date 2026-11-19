import copy
import itertools
import math
import time

import numpy as np
from scipy.special import binom, gammaln
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
        n_masks: Number of coalitions to sample per bag
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
        RISE attribution score for each patch
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
            print(masks.shape, preds.shape, len(pi_z))
            lr.fit(masks, preds, sample_weight=pi_z)

            # Extract scores
            coefs = torch.tensor(lr.coef_).squeeze()
            scores.append(coefs)
    return scores


def shap(dataset, model, n_masks, device):
    """Performs LeverageSHAP.
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
    scores = []

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for _, bag in enumerate(tqdm(dataloader)):
            masks = []
            preds = []
            pi_z = []

            X = bag["X"].to(device)
            num_patches = X.shape[1]

            # define the correct number of samples
            nsamples = min(2 * num_patches + 2**11, 20000)
            num_samples_left = nsamples
            num_subset_sizes = int(np.ceil((num_patches - 1) / 2))
            num_paired_subset_sizes = int(np.floor((num_patches - 1) / 2))
            added = 0
            num_full_subsets = 0
            weights = np.array(
                [
                    (num_patches - 1) / (i * (num_patches - i))
                    for i in range(1, num_subset_sizes + 1)
                ]
            )
            weights[:num_paired_subset_sizes] *= 2
            weights /= np.sum(weights)
            remaining_weights = copy.copy(weights)

            # Add full bag, as kernel would be infinity
            full_mask = torch.ones((num_patches,))
            full_pred = model(X).cpu()
            # masks.append(full_mask.cpu())
            # preds.append(full_pred.cpu())
            # pi_z.append(math.log(1e6))

            # Add empty bag with null embedding
            X_subset = torch.zeros_like(X)
            empty_mask = torch.zeros((num_patches,))
            empty_pred = model(X_subset).cpu()
            # masks.append(empty_mask.cpu())
            # preds.append(empty_pred.cpu())
            # pi_z.append(math.log(1e6))

            for subset_size in range(1, num_subset_sizes + 1):
                bin = binom(num_patches, subset_size)
                to_add = bin
                if subset_size <= num_paired_subset_sizes:
                    to_add *= 2
                if num_samples_left * remaining_weights[subset_size - 1] / to_add >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= to_add
                    added += to_add

                    # rescale weights so the remaining coalitions have weights adding to 1
                    if remaining_weights[subset_size - 1] < 1:
                        remaining_weights /= 1 - remaining_weights[subset_size - 1]
                    w = weights[subset_size - 1] / bin
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2
                    for inds in itertools.combinations(np.arange(num_patches), subset_size):
                        mask = torch.zeros(num_patches, dtype=torch.bool)
                        mask[torch.tensor(inds, dtype=torch.long)] = True

                        # Add sample
                        X_subset = X[:, mask]
                        pred = model(X_subset).cpu()
                        masks.append(mask)
                        preds.append(pred)
                        pi_z.append(w)

                        if subset_size <= num_paired_subset_sizes:
                            mask = ~mask
                            # Add inverted sample
                            X_subset = X[:, mask]
                            pred = model(X_subset).cpu()
                            masks.append(mask)
                            preds.append(pred)
                            pi_z.append(w)
                else:
                    break

            # Remaining samples when a full subset does not fit anymore
            if num_full_subsets != num_subset_sizes:
                remaining_weights = copy.copy(weights)
                remaining_weights[:num_paired_subset_sizes] /= 2
                remaining_weights = remaining_weights[num_full_subsets:]
                remaining_weights /= np.sum(remaining_weights)
                # Generate more cardinalities to handle used samples
                indices = np.random.choice(len(remaining_weights), size=4 * num_samples_left, p=remaining_weights)
                ind_pos = 0
                used_masks = {}
                unfixed_weights = {}
                while num_samples_left > 0 and ind_pos < len(indices):
                    mask = torch.zeros(num_patches)
                    ind = indices[ind_pos]
                    ind_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    inds = torch.randperm(num_patches)[:subset_size]
                    mask[torch.tensor(inds, dtype=torch.int64)] = 1

                    mask_tuple = tuple(mask.tolist())
                    if mask_tuple not in used_masks:
                        used_masks[mask_tuple] = added
                        unfixed_weights[used_masks[mask_tuple]] = 1
                        added += 1
                        num_samples_left -= 1
                    else:
                        unfixed_weights[used_masks[mask_tuple]] += 1

                    if num_samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask = 1 - mask
                        mask_tuple = tuple(mask.tolist())
                        if mask_tuple not in used_masks:
                            used_masks[mask_tuple] = added
                            unfixed_weights[used_masks[mask_tuple]] = 1
                            added += 1
                            num_samples_left -= 1
                        else:
                            unfixed_weights[used_masks[mask_tuple]] += 1

                weight_left = np.sum(weights[num_full_subsets:])
                unfixed_weights_sum = sum(unfixed_weights.values())
                unfixed_weights = {k: v * weight_left / unfixed_weights_sum for k, v in unfixed_weights.items()}
                for mask, pos in used_masks.items():
                    X_subset = X[:, torch.tensor(mask, dtype=torch.bool)]
                    pred = model(X_subset).cpu()
                    weight = unfixed_weights[pos]
                    pi_z.append(weight)
                    masks.append(torch.tensor(mask, dtype=torch.float32))
                    preds.append(pred)

            preds = [pred - empty_pred for pred in preds]
            attribution_sum = full_pred - empty_pred
            removed_features = [mask[-1] for mask in masks]
            y = [pred - attribution_sum * removed_feature for pred, removed_feature in zip(preds, removed_features)]
            modified_mask = [mask[:-1] - removed_feature for mask, removed_feature in zip(masks, removed_features)]

            # Fit model
            lr = LinearRegression(fit_intercept=False)
            modified_mask = torch.stack(modified_mask).numpy()
            y = torch.stack(y).numpy().squeeze()

            # print("Masks:", len(masks))
            # print("Modified mask:", modified_mask.shape)
            # print("Y:", y.shape)
            # print("Weights:", len(pi_z))
            # print("Patches:", num_patches)
            lr.fit(modified_mask, y, sample_weight=pi_z)

            # Extract scores
            coefs = torch.tensor(lr.coef_).squeeze()
            last_coef = attribution_sum - coefs.sum()
            coefs = torch.cat([coefs, last_coef.reshape(1)])
            scores.append(coefs)

            # for _ in range(n_masks):
            #     #
            #     # Sample patches
            #     num_patches_to_sample = torch.randint(
            #         low=1, high=num_patches, size=(1,)
            #     ).item()
            #     mask = torch.randperm(num_patches)[:num_patches_to_sample]
            #     bin_mask = torch.zeros((num_patches,))
            #     bin_mask[mask] = 1

            #     # Compute weight
            #     log_comb = (
            #         ln_n_fac
            #         - gammaln(num_patches_to_sample + 1)
            #         - gammaln(num_patches - num_patches_to_sample + 1)
            #     )
            #     weight = math.log(num_patches - 1) - (
            #         (log_comb)
            #         + math.log(num_patches_to_sample)
            #         + math.log(num_patches - num_patches_to_sample)
            #     )

            #     X_subset = X[:, mask]

            #     # Predict logits
            #     pred = model(X_subset).cpu()

            #     masks.append(bin_mask.cpu())
            #     preds.append(pred.cpu())
            #     pi_z.append(weight)

            # # Fit model
            # lr = LinearRegression(fit_intercept=True)
            # # t1 = time.time()
            # masks = torch.stack(masks).numpy()
            # preds = torch.stack(preds).numpy().squeeze()
            # # t3 = time.time()
            # pi_max = max(pi_z)
            # pi_z = [math.exp(pi - pi_max) for pi in pi_z]
            # # t4 = time.time()
            # # print(t4-t3, "secs to exponentiate")
            # lr.fit(masks, preds, sample_weight=pi_z)
            # # t2 = time.time()
            # # print(t2-t1)

            # # Extract scores
            # coefs = torch.tensor(lr.coef_).squeeze()
            # scores.append(coefs)
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
