import warnings

import torch
from torchmil.datasets import CAMELYON16MILDataset
from torchmil.models import ABMIL
from torch.utils.data import DataLoader
from torchmil.data import collate_fn

from src.training import evaluate, train
from src.xai_eval.numerical_metrics import compute_numerical_metrics, print_numerical_metrics
from src.xai_eval.plot import plot_all_numerical_metrics
from src.xai_eval.rsrg import rsrg
from src.xai_eval.srg import srg

warnings.filterwarnings("ignore")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 10

    train_dataset = CAMELYON16MILDataset(root="data", features="UNI")
    train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )

    test_dataset = CAMELYON16MILDataset(root="data", features="UNI", partition="test")
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    model = ABMIL(in_shape=(1024,), criterion=torch.nn.BCEWithLogitsLoss()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    train(model, optimizer, criterion, train_dataloader, epochs, device)
    _, _, _, patch_labels, attention_scores = evaluate(
        model, criterion, test_dataloader, device
    )

    explanation_metrics = compute_numerical_metrics(attention_scores, patch_labels)
    plot_all_numerical_metrics(explanation_metrics, "results/plots/numerical_metrics")
    print_numerical_metrics(explanation_metrics)

    # srg_metrics = srg(attention_scores, model, device, num_bins=100)
    # print("Ascending:", srg_metrics["ascending"])
    # print("Descending:", srg_metrics["descending"])
    # print("SRG=", srg_metrics["srg"], sep="")

    # rsrg_metrics = rsrg(model, attention_scores, 3, test_dataset, device, plot=True)
    # print("R-LIF:", rsrg_metrics["R-LIF"])
    # print("R-MIF:", rsrg_metrics["R-MIF"])
    # print("RSRG=", rsrg_metrics["RSRG"], sep="")

    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    main()
