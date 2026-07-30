import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmil.data import collate_fn
from torchmil.nn.attention.attention_pool import AttentionPool
from tqdm import tqdm

activations = dict()
execution_order = []


def save_activations(name):
    """Generates the hooks for the layers

    Args:
        name: Name of the layer
    """
    def hook(module, input, output):
        if (name, module) not in execution_order:
            execution_order.append((name, module))
        if isinstance(output, tuple):
            activations[name] = {
                "input": input[0].detach().cpu(),
                "output": output[0].detach().cpu(),
            }
        else:
            activations[name] = {
                "input": input[0].detach().cpu(),
                "output": output.detach().cpu(),
            }

    return hook


def register_hooks(model):
    for name, layer in model.named_modules():
        layer.register_forward_hook(save_activations(name))


def modify_weights(weights, gamma):
    """Modifies the weights with the gamma-rule. If gamma = 0, this is the epsilon-rule

    Args:
        weights: The weights
        gamma: The weight of the gamma rule, adds the non-zero weight scaled by gamma

    Returns:
        The modified weights
    """
    zeros = torch.zeros_like(weights)
    non_negative, _ = torch.max(torch.cat((zeros, weights), dim=0), dim=0, keepdim=True)
    return weights + gamma * non_negative


def lrp(dataset, model, device):
    scores = []
    register_hooks(model)
    eps = 1e-7
    chunk_size = 256
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    for bag in tqdm(dataloader):
        with torch.no_grad():
            r, attention = model(bag["X"].to(device), return_att=True)
        r = r.cpu()
        attention = attention.cpu()
        for name, layer in execution_order[::-1]:
            if isinstance(layer, nn.Linear):
                w = layer.weight.T.cpu()  # (d_l, d_l+1)
                w = modify_weights(w, 0)  # gamma rule
                a = activations[name]["input"].cpu()
                a_unsqueezed = a.unsqueeze(-1)  # (batch_dim, patch_dim, d_l, 1)
                r_unsqueezed = r.unsqueeze(-2)  # (batch_dim, patch_dim, 1, d_l+1)

                r_new = []
                for chunk in range(0, a.shape[1], chunk_size):
                    a_start = chunk
                    a_end = chunk + chunk_size
                    patch_dim_exists = a_unsqueezed.ndim > 3
                    if patch_dim_exists:
                        a_chunked = a_unsqueezed[:, a_start:a_end]
                    else:
                        a_chunked = a_unsqueezed[:]
                    if r_unsqueezed.ndim > 3:  # patch dimension exists
                        r_chunked = r_unsqueezed[:, a_start:a_end]
                    else:
                        r_chunked = r_unsqueezed[:]
                    z = a_chunked * w  # (batch_dim, patch_dim, d_l, d_l+1)
                    denom = eps + torch.sum(
                        z, -2, keepdim=True
                    )  # (batch_dim, patch_dim, 1, d_l+1)
                    # print("denom", denom.shape)
                    chunk_r = torch.sum(z / denom * r_chunked, -1)
                    r_new.append(chunk_r)
                    if not patch_dim_exists:
                        break
                r = torch.cat(r_new, dim=1)
            if isinstance(layer, nn.Tanh):
                pass
            if isinstance(layer, AttentionPool):
                h = bag["X"]  # (batch_dim, patch_dim, feature_dim)
                a = torch.softmax(attention, dim=1).cpu()  # (batch_dim, patch_dim)
                z = a.unsqueeze(-1) * h  # (batch_dim, patch_dim, feature_dim)
                denom = (
                    torch.sum(z, dim=1, keepdim=True) + eps
                )  # (batch_dim, 1, feature_dim)
                r = z / denom * r
        # scores (1, patch_size, embedding_size)
        scores.append(r.sum(-1).flatten().detach().cpu())
    return scores
