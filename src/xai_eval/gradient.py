import torch
from torch.utils.data import DataLoader
from torchmil.data import collate_fn
from tqdm import tqdm


def gradient_x_input(dataset, model, device):
    """Gradient X Input

    Args:
        dataset: The dataset
        model: The trained model
        device: Device used for computation

    Returns:
        Return attribution scores accoding to GxI
    """
    scores = []
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for bag in tqdm(dataloader):
        X = bag["X"].to(device)
        X.requires_grad_(True)
        model.zero_grad()
        X.grad = None
        pred = model(X).squeeze()
        pred.backward()
        grad = X.grad
        grad_x_input = (grad * X).sum(dim=-1).detach().cpu().squeeze()
        scores.append(grad_x_input)
    return scores


def integrated_gradients(dataset, model, device, steps=50):
    """Integrated Gradients
    Approximated by the mean of gradients on the line between the baseline 0 and the actual X

        Args:
            dataset: The dataset
            model: The trained model
            device: Device used for computation
            steps: Steps to approximate integral of gradient with. Defaults to 50.

        Returns:
            Return attribution scores accoding to GxI
    """
    scores = []
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    for bag in tqdm(dataloader):
        X = bag["X"].to(device)
        X.requires_grad_(True)

        alphas = torch.linspace(0, 1, steps=steps)

        sum_term = torch.zeros_like(X)
        for alpha in alphas:
            # Compute element at fraction alpha_i of line from baseline to X
            X_alpha = alpha * X
            X_alpha = X_alpha.to(device)
            model.zero_grad()
            X.grad = None
            pred_alpha = model(X_alpha).squeeze()
            pred_alpha.backward()
            sum_term += X.grad

        # Compute approximation of the integral
        IG_i = 1 / len(alphas) * X.squeeze() * sum_term.squeeze()
        IG_i_sum = IG_i.sum(dim=-1)
        scores.append(IG_i_sum.detach().cpu())
    return scores
