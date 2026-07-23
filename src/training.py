import numpy as np
import torch
from tqdm import tqdm


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
