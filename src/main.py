import warnings

import pandas as pd
import torch
from torchmil.datasets import CAMELYON16MILDataset
from torchmil.models import ABMIL
from torch.utils.data import DataLoader
from torchmil.data import collate_fn

from src.training import evaluate, train
from src.utils import shape_test
from src.xai_eval.gradient import gradient_x_input, integrated_gradients
from src.xai_eval.lrp import lrp
from src.xai_eval.numerical_metrics import (
    compute_numerical_metrics,
    print_numerical_metrics,
)
from src.xai_eval.perturbation import (
    combined_perturbation,
    one_removed_perturbation,
    milli,
    single_perturbation,
)
from src.xai_eval.plot import plot_all_numerical_metrics
from src.xai_eval.rsrg import rsrg
from src.xai_eval.srg import srg

warnings.filterwarnings("ignore")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ABMIL(in_shape=(1024,), criterion=torch.nn.BCEWithLogitsLoss()).to(device)
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    test_dataset = CAMELYON16MILDataset(root="data", features="UNI", partition="test")
    slide_labels = torch.stack([slide["Y"] for slide in test_dataset])

    lrp_scores = torch.load("results/scores/lrp_scores.pt")
    patch_labels = torch.load("results/scores/patch_labels.pt")
    attention_scores = torch.load("results/scores/attention_scores.pt")
    perturbation_scores = torch.load("results/scores/perturbation_scores.pt")
    milli_scores = torch.load("results/scores/milli_scores.pt")
    gradient_x_input_scores = torch.load("results/scores/gxi_scores.pt")
    ig_scores = torch.load("results/scores/ig_scores.pt")

    scores = {
        "milli": milli_scores,
        "attention": attention_scores,
        "single_perturbation": perturbation_scores["single"],
        "one_removed_perturbation": perturbation_scores["one_removed"],
        "combined_perturbation": perturbation_scores["combined"],
        "gradient_x_input": gradient_x_input_scores,
        "integrated_gradients": ig_scores,
        "lrp": lrp_scores,
    }

    shape_test(test_dataset, scores)

    # for i, bag in enumerate(test_dataset):
    #     for method in ["single", "one_removed", "combined"]:
    #         print(method)
    #         scores = perturbation_scores[method][i]
    #         print(scores.shape)
    #         labels = patch_labels[i]

    #         topk = torch.argsort(scores, descending=True)[:20]
    #         print("\nslide label", bag["Y"])
    #         print(labels[topk])
    #         topk_precision = labels[topk].float().mean()
    #         print(topk_precision)

    metrics_table = {}

    for method, method_scores in scores.items():
        print(method)
        numerical_metrics = compute_numerical_metrics(
            method_scores, patch_labels, slide_ground_truth=slide_labels
        )
        plot_all_numerical_metrics(
            numerical_metrics, f"results/plots/numerical_metrics/{method}"
        )
        metrics_table[method] = {
            "AUROC": numerical_metrics["auroc"],
            "AP": numerical_metrics["average_precision"],
            "Micro-AUROC Pos": numerical_metrics["ground_truth_positive_slides"][
                "micro"
            ]["auroc"],
            "Micro-AP Pos": numerical_metrics["ground_truth_positive_slides"]["micro"][
                "ap"
            ],
            "Macro-AUROC Pos": numerical_metrics["ground_truth_positive_slides"][
                "macro"
            ]["auroc"],
            "Macro-AP Pos": numerical_metrics["ground_truth_positive_slides"]["macro"][
                "ap"
            ],
            "Pearson": numerical_metrics["normalized_correlation"],
            "Macro-Pearson Pos": numerical_metrics["ground_truth_positive_slides"][
                "macro"
            ]["pearson"],
            "Macro-Pearson Pos": numerical_metrics["ground_truth_positive_slides"][
                "macro"
            ]["pearson"],
        }
        if method == "integrated_gradients":
            print("tumor")
            print(
                numerical_metrics["scores"]["tumor"].min(),
                numerical_metrics["scores"]["tumor"].max(),
            )
            print("normal")
            print(
                numerical_metrics["scores"]["normal"].min(),
                numerical_metrics["scores"]["normal"].max(),
            )
            scores_sorted, idx_sorted = torch.sort(numerical_metrics["scores"]["tumor"])
            print(scores_sorted[:10])
            print(numerical_metrics["positive_labels"][idx_sorted[:10]])
            print(scores_sorted[-10:])
            print(numerical_metrics["positive_labels"][idx_sorted[-10:]])

    scores = torch.cat(gradient_x_input_scores)

    print(scores.min())
    print(scores.max())
    print(scores.mean())
    print(scores.std())
    for q in [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1]:
        print(q, torch.quantile(scores, q))

    df = pd.DataFrame(metrics_table).T
    print(df)
    torch.save(metrics_table, "results/scores/metrics.pt")

    # srg_metrics = srg(milli_scores, model, device, num_bins=100)
    # print("Ascending:", srg_metrics["ascending"])
    # print("Descending:", srg_metrics["descending"])
    # print("SRG=", srg_metrics["srg"], sep="")

    # rsrg_metrics = rsrg(model, attention_scores, 3, test_dataset, device, plot=True)
    # print("R-LIF:", rsrg_metrics["R-LIF"])
    # print("R-MIF:", rsrg_metrics["R-MIF"])
    # print("RSRG=", rsrg_metrics["RSRG"], sep="")


if __name__ == "__main__":
    main()
