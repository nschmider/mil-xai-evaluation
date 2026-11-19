import copy
import itertools
import math
import time

import numpy as np
from scipy.special import binom, gammaln
from sklearn.linear_model import LinearRegression
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmil.data import collate_fn
from torchmil.models import ABMIL
from tqdm import tqdm

from src.models import ShapPredictor


def train(
    model,
    optimizer,
    criterion,
    dataset,
    max_epochs,
    device,
    early_stopping=True,
    validation_dataset=None,
    patience=3,
):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    model.train()
    losses = []
    accuracies = []

    plateau_epochs = 0
    best_model_state = None
    best_loss = float("inf")

    for e in range(max_epochs):
        loss_sum = 0
        accuracy = 0

        pbar = tqdm(dataloader, desc=f"[Epoch {e+1}] Train : ")
        for i, batch in enumerate(pbar):
            X = batch["X"].to(device)
            Y = batch["Y"].to(device)
            mask = batch["mask"].to(device)
            Y_pred = model(X, mask)

            loss = criterion(Y_pred, Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            accuracy += ((Y_pred > 0) == Y).float().mean()

            pbar.set_postfix(
                {
                    "loss": f"{loss_sum / (i+1):.3f}",
                    "accuracy": f"{accuracy / (i+1):.3f}",
                }
            )

        losses.append(loss_sum / len(dataloader))
        accuracies.append((accuracy / len(dataloader)).item())

        # Early stopping check
        if early_stopping:
            validation_loss = evaluate(model, criterion, validation_dataset, device)[
                "losses"
            ]
            validation_loss = sum(validation_loss) / len(validation_loss)
            if validation_loss >= best_loss:
                plateau_epochs += 1
                if plateau_epochs > patience:
                    model.load_state_dict(best_model_state)
                    break
            else:
                plateau_epochs = 0
                best_loss = loss
                best_model_state = copy.deepcopy(model.state_dict())
            print(validation_loss)
            print(plateau_epochs)

    return losses, accuracies


def evaluate(model, criterion, dataset, device):
    t1 = time.time()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model.eval()
    losses = []
    accuracies = []
    all_attention_scores = []
    all_patch_labels = []
    preds = []

    with torch.no_grad():
        loss_sum = 0
        accuracy = 0

        pbar = tqdm(dataloader, desc=f"Evaluate : ")
        for i, batch in enumerate(pbar):
            X = batch["X"].to(device)
            Y = batch["Y"].to(device)
            patch_label = batch["y_inst"].squeeze().to(device)
            mask = batch["mask"].to(device)
            Y_pred, attention_scores = model(X, mask, return_att=True)

            loss = criterion(Y_pred, Y.float())
            loss_sum += loss.cpu().item()
            accuracy += ((Y_pred > 0) == Y).float().mean()

            attention_scores = torch.squeeze(attention_scores, dim=0)
            all_attention_scores.append(attention_scores.cpu())
            all_patch_labels.append(patch_label.cpu())
            preds.append(Y_pred.cpu())

            pbar.set_postfix(
                {
                    "loss": f"{loss_sum / (i+1):.3f}",
                    "accuracy": f"{accuracy / (i+1):.3f}",
                }
            )

        losses.append(loss_sum / len(dataloader))
        accuracies.append((accuracy / len(dataloader)).item())
        preds = np.array(preds).flatten()
    t2 = time.time()

    return {
        "losses": losses,
        "accuracies": accuracies,
        "preds": preds,
        "patch_labels": all_patch_labels,
        "attention_scores": all_attention_scores,
        "time": t2 - t1,
    }


def train_shap_predictor_old(model, dataset, n_masks, device):
    """Performs KernelSHAP.
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

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    xs = []
    scores = []
    with torch.no_grad():
        for _, bag in enumerate(tqdm(dataloader)):
            masks = []
            preds = []
            pi_z = []

            # Add full bag, as kernel would be infinity
            X = bag["X"].to(device)
            num_patches = X.shape[1]
            full_mask = torch.ones((num_patches,))
            full_pred = model(X).cpu()
            masks.append(full_mask.cpu())
            preds.append(full_pred.cpu())
            pi_z.append(math.log(1e6))

            # Add empty bag with null embedding
            X_subset = torch.zeros_like(X)
            empty_mask = torch.zeros((num_patches,))
            empty_pred = model(X_subset).cpu()
            masks.append(empty_mask.cpu())
            preds.append(empty_pred.cpu())
            pi_z.append(math.log(1e6))

            ln_n_fac = gammaln(num_patches + 1)

            for _ in range(n_masks):
                # Sample patches
                num_patches_to_sample = torch.randint(
                    low=1, high=num_patches, size=(1,)
                ).item()
                mask = torch.randperm(num_patches)[:num_patches_to_sample]
                bin_mask = torch.zeros((num_patches,))
                bin_mask[mask] = 1

                # Compute weight
                log_comb = (
                    ln_n_fac
                    - gammaln(num_patches_to_sample + 1)
                    - gammaln(num_patches - num_patches_to_sample + 1)
                )
                weight = math.log(num_patches - 1) - (
                    (log_comb)
                    + math.log(num_patches_to_sample)
                    + math.log(num_patches - num_patches_to_sample)
                )

                X_subset = X[:, mask]

                # Predict logits
                pred = model(X_subset).cpu()

                masks.append(bin_mask.cpu())
                preds.append(pred.cpu())
                pi_z.append(weight)

            # Fit model
            lr = LinearRegression(fit_intercept=True)
            masks = torch.stack(masks).numpy()
            preds = torch.stack(preds).numpy().squeeze()
            pi_max = max(pi_z)
            pi_z = [math.exp(pi - pi_max) for pi in pi_z]
            lr.fit(masks, preds, sample_weight=pi_z)

            # Extract scores
            coefs = torch.tensor(lr.coef_).squeeze()
            scores.append(coefs)
            xs.append(X.squeeze(0))

    # Train model on Shapley Values
    shap_predictor = fit_shap_predictor(xs, scores, device)
    return shap_predictor


def train_shap_predictor(model, dataset, n_masks, device):
    """Performs KernelSHAP.
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

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    xs = []
    scores = []
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
            xs.append(X.squeeze(0).detach().cpu())
            scores.append(coefs.detach().cpu().reshape(-1))
    # Train model on Shapley Values
    shap_predictor = fit_shap_predictor(xs, scores, device)
    return shap_predictor


def fit_shap_predictor(xs, scores, device):
    shap_predictor = ShapPredictor().to(device)
    shap_predictor.train()
    criterion = MSELoss().to(device)
    optimizer = Adam(shap_predictor.parameters(), lr=1e-4)
    flattened_xs = torch.cat(xs, dim=0).cpu()
    flattened_scores = torch.cat(scores, dim=0)
    dataset = TensorDataset(flattened_xs, flattened_scores)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    for x, y in tqdm(dataloader):
        pred = shap_predictor(x.to(device))
        loss = criterion(pred, y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return shap_predictor


def eval_shap_predictor(shap_predictor, dataset, device):
    shap_predictor.eval()
    shap_predictor = shap_predictor.to(device)
    scores = []
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    with torch.no_grad():
        for bag in tqdm(dataloader):
            X = bag["X"].squeeze(0).to(device)
            patch_scores = shap_predictor(X).squeeze()
            scores.append(patch_scores.detach().cpu())
    return scores
