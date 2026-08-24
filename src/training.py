import math

import numpy as np
from scipy.special import gammaln
from sklearn.linear_model import LinearRegression
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchmil.data import collate_fn
from torchmil.models import ABMIL
from tqdm import tqdm

from src.models import ShapPredictor


def train(model, optimizer, criterion, dataloader, epochs, device):
    model.train()
    losses = []
    accuracies = []
    for e in range(epochs):
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

    return losses, accuracies


def evaluate(model, criterion, dataloader, device):
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

    return losses, accuracies, preds, all_patch_labels, all_attention_scores


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
                log_comb = ln_n_fac - gammaln(num_patches_to_sample + 1) - gammaln(num_patches - num_patches_to_sample + 1)
                weight = math.log(num_patches - 1) - (
                    (log_comb) + math.log(num_patches_to_sample)
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
