from matplotlib import pyplot as plt
import numpy as np
import random
import torch
from torchmetrics.classification import BinaryAccuracy
from torchmil.datasets import CAMELYON16MILDataset
from torchmil.models import ABMIL
from torchmil.utils import Trainer
from torch.utils.data import DataLoader
from torchmil.data import collate_fn
import warnings

from src.training import train

warnings.filterwarnings("ignore")


class Logger:
    def __init__(self):
        self.history = []

    def log(self, metrics):
        self.history.append(metrics.copy())


def plot_trainers(device):
    """Plots training accuracy and loss from TorchMIL trainer against self-implemented trainer

    Args:
        device: Device used for computation
    """
    # Fix seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    epochs = 10
    dataset = CAMELYON16MILDataset(root="data", features="UNI")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Train with my method
    model = ABMIL(in_shape=(1024,), criterion=torch.nn.BCEWithLogitsLoss()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    losses_own, accuracies_own = train(model, optimizer, criterion, dataloader, epochs)

    # Train with trainer
    model = ABMIL(in_shape=(1024,), criterion=torch.nn.BCEWithLogitsLoss()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics_dict = {"accuracy": BinaryAccuracy().to(device)}
    logger = Logger()
    trainer = Trainer(
        model, optimizer, metrics_dict=metrics_dict, device=device, logger=logger
    )
    trainer.train(train_dataloader=dataloader, max_epochs=epochs)
    losses_trainer = [
        epoch["train/loss"] for epoch in logger.history if "train/loss" in epoch.keys()
    ]
    accuracies_trainer = [
        epoch["train/accuracy"]
        for epoch in logger.history
        if "train/accuracy" in epoch.keys()
    ]

    # Plot losses
    plt.figure()
    plt.plot(range(1, epochs + 1), losses_own, label="My loss", color="green")
    plt.plot(range(1, epochs + 1), losses_trainer, label="Trainer loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/plots/comparison_trainers_loss.png")

    # Plot accuracies
    plt.figure()
    plt.plot(range(1, epochs + 1), accuracies_own, label="My accuracy", color="green")
    plt.plot(
        range(1, epochs + 1), accuracies_trainer, label="Trainer accuracy", color="blue"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("results/plots/comparison_trainers_accuracy.png")
    plt.close()
