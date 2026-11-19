import os
import time
import warnings

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import torch
from torchmil.datasets import CAMELYON16MILDataset
from torchmil.models import ABMIL
from torch.utils.data import DataLoader, random_split
from torchmil.data import collate_fn

from src.models import ShapPredictor
from src.training import evaluate, train, train_shap_predictor, eval_shap_predictor
from src.utils import set_seed, shape_test
from src.visualize import compute_slides
from src.xai_eval.gradient import (
    gradient_x_input,
    integrated_gradients,
    squared_gradients,
)
from src.xai_eval.lrp import lrp
from src.xai_eval.numerical_metrics import (
    compute_numerical_metrics,
    print_numerical_metrics,
)
from src.xai_eval.perturbation import (
    combined_perturbation,
    one_removed_perturbation,
    lime,
    milli,
    rise,
    single_perturbation,
    shap,
)
from src.xai_eval.plot import plot_all_numerical_metrics
from src.xai_eval.rsrg import rsrg
from src.xai_eval.srg import srg

warnings.filterwarnings("ignore")


def main():
    methods = [
        # ("attention_scores.pt", "attention", "attention"),
        ("gradient_x_input_scores.pt", "gradient_x_input", "gradient x input"),
        ("squared_gradients_scores.pt", "squared_gradients", "squared gradients"),
        ("one_removed_scores.pt", "one_removed_perturbation", "one-removed perturbation"),
        ("combined_perturbation_scores.pt", "combined_perturbation", "combined perturbation"),
        ("ig_scores.pt", "integrated_gradients", "integrated gradients"),
        ("lime_scores.pt", "lime", "LIME"),
        ("rise_scores.pt", "rise", "RISE"),
        ("lrp_scores.pt", "lrp", "LRP"),
        ("milli_scores.pt", "milli", "MILLI"),
        ("shap_scores.pt", "shap", "SHAP"),
        ("single_perturbation_scores.pt", "single_perturbation", "single perturbation"),
    ]

    # seed = 0
    # for filename, plot_dir, description in methods:
    #     scores = torch.load(f"results/scores/{seed}/{filename}")
    #     base_plot_dir = "results/plots/visualized_scores"
    #     output_dir = os.path.join(base_plot_dir, plot_dir)
    #     os.makedirs(output_dir, exist_ok=True)

    #     elapsed = compute_slides(
    #         scores,
    #         output_dir,
    #         alpha=0.5,
    #         gt=False
    #     )
    # # return
    #     # print(elapsed, f"needed for {description}")
    runtimes = []
    runtimes2 = []

    # for filename, plot_dir, description in methods:
    #     for seed in range(5):
    #         set_seed(seed)
    #         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #         model = ABMIL(in_shape=(1024,), criterion=torch.nn.BCEWithLogitsLoss()).to(
    #             device
    #         )
    #         criterion = torch.nn.BCEWithLogitsLoss()
    #         train_dataset = CAMELYON16MILDataset(
    #             root="data", features="UNI", partition="train"
    #         )
    #         test_dataset = CAMELYON16MILDataset(
    #             root="data", features="UNI", partition="test"
    #         )
    #         train_dataset, validation_dataset = random_split(
    #             train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
    #         )
    #         scores = torch.load(f"results/scores/{seed}/{filename}")

    #         print(plot_dir)
    #         rsrg_metrics = rsrg(
    #             model,
    #             scores,
    #             10,
    #             test_dataset,
    #             device,
    #             plot=False,
    #             method=f"shap/{seed}",
    #             step=20,
    #         )
    #         print("R-LIF:", rsrg_metrics["R-LIF"])
    #         print("R-MIF:", rsrg_metrics["R-MIF"])
    #         print("RSRG=", rsrg_metrics["RSRG"], sep="")
    #         print("Time needed for computation", rsrg_metrics["time"])
    #         print("seed", seed)
    #         print("Runtimes", runtimes)
    all_scores = {}
    metrics_tables = []
    for seed in range(1):
        # seed = 2
        set_seed(seed)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ABMIL(in_shape=(1024,), criterion=torch.nn.BCEWithLogitsLoss()).to(
            device
        )
        criterion = torch.nn.BCEWithLogitsLoss()
        train_dataset = CAMELYON16MILDataset(
            root="data", features="UNI", partition="train"
        )
        test_dataset = CAMELYON16MILDataset(
            root="data", features="UNI", partition="test"
        )
        train_dataset, validation_dataset = random_split(
            train_dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
        # losses, accuracies = train(
        #     model,
        #     optimizer=torch.optim.Adam(model.parameters()),
        #     criterion=criterion,
        #     dataset=train_dataset,
        #     max_epochs=100,
        #     device=device,
        #     early_stopping=True,
        #     validation_dataset=validation_dataset,
        #     patience=3,
        # )
        # evaluation_metrics = evaluate(model, criterion, test_dataset, device)
        # attention_scores = evaluation_metrics["attention_scores"]
        # evaluation_time = evaluation_metrics["time"]
        # torch.save(model.state_dict(), f"models/{seed}/model.pth")
        # torch.save(attention_scores, f"results/scores/{seed}/attention_scores.pt")
        model.load_state_dict(torch.load(f"models/{seed}/model.pth", weights_only=True))
        model.eval()
        # slide_labels = torch.stack([slide["Y"] for slide in test_dataset])
        # patch_labels = [slide["y_inst"] for slide in test_dataset]
        # scores = torch.load(f"results/scores/{seed}/attention_scores.pt")
        # compute_slides(scores, "results/plots/visualized_scores/ground_truth", alpha=0.5, gt=patch_labels)
        # return

    #     scores = {}
    #     for filename, plot_dir, description in methods:
    #         print(description)
    #         print(seed)
    #         cur_scores = torch.load(f"results/scores/{seed}/{filename}")
    #         srg_score = srg(cur_scores, model, device, num_bins=100, plot=False)["srg"]
    #         print(srg_score)
    #         # SRG-Wert speichern
    #         if isinstance(srg_score, torch.Tensor):
    #             srg_score = srg_score.detach().cpu().item()

    #         scores[description] = float(srg_score)

    #     # Ergebnisse dieses Seeds speichern
    #     all_scores[seed] = scores


    # # ---------------------------------
    # # Tabelle 1: Einzelne Seeds
    # # ---------------------------------

    # import pandas as pd
    # import numpy as np

    # df_seeds = pd.DataFrame(all_scores)
    # df_seeds.index.name = "Method"
    # df_seeds.columns = [f"Seed {seed}" for seed in df_seeds.columns]

    # print("\nSRG scores per seed:")
    # print(df_seeds.round(4).to_string())

    # print("\nLaTeX:")
    # print(df_seeds.to_latex(
    #     float_format="%.4f",
    #     column_format="l" + "c" * 5
    # ))


    # # ---------------------------------
    # # Tabelle 2: Mean und Std
    # # ---------------------------------

    # summary = pd.DataFrame({
    #     "Mean": df_seeds.mean(axis=1),
    #     "Std": df_seeds.std(axis=1, ddof=1)
    # })

    # print("\nMean and Std across seeds:")
    # print(summary.round(4).to_string())

    # print("\nLaTeX:")
    # print(summary.to_latex(
    #     float_format="%.4f",
    #     column_format="lcc"
    # ))


    # # ---------------------------------
    # # Tabelle 3: Mean ± Std
    # # ---------------------------------

    # summary_latex = pd.DataFrame({
    #     "Mean ± Std": [
    #         f"${mean:.4f} \\pm {std:.4f}$"
    #         for mean, std in zip(summary["Mean"], summary["Std"])
    #     ]
    # }, index=summary.index)

    # print("\nLaTeX (Mean ± Std):")
    # print(summary_latex.to_latex(
    #     escape=False,
    #     header=True,
    #     column_format="lc"
    # ))
    # return
        

        # trained_shap_scores_1k = torch.load("results/scores/trained_shap_scores")
        # shap_scores_1k = torch.load("results/scores/shap_scores")
        # shap_predictor = train_shap_predictor(model, train_dataset, 2000, device)
        # torch.save(shap_predictor.state_dict(), "models/shap_predictor.pth")
        # trained_shap_scores = eval_shap_predictor(shap_predictor, test_dataset, device)
        # torch.save(trained_shap_scores, "results/scores/trained_shap_scores")
        # shap_scores = shap(test_dataset, model, 2000, device)
        # torch.save(shap_scores, "results/scores/shap_scores")
        # shap_scores = torch.load("results/scores/shap_scores.pt")
        # lime_scores = torch.load("results/scores/lime_scores.pt")
        # rise_scores = torch.load("results/scores/rise_scores.pt")
        # squared_gradients_scores = squared_gradients(test_dataset, model, device)
        # torch.save(squared_gradients_scores, "results/scores/squared_gradients_scores.pt")
        # squared_gradients_scores = torch.load(
        #     "results/scores/squared_gradients_scores.pt"
        # )
        # lrp_scores = lrp(test_dataset, model, device)
        # torch.save(lrp_scores, "results/scores/lrp_scores.pt")
        # lrp_scores = torch.load("results/scores/lrp_scores.pt")
        # patch_labels = torch.load("results/scores/patch_labels.pt")
        # attention_scores = torch.load("results/scores/attention_scores.pt")
        # perturbation_scores = torch.load("results/scores/perturbation_scores.pt")
        # single_perturbation_scores = perturbation_scores["single"]
        # single_perturbation_scores = torch.load(
        #     "results/scores/single_perturbation_scores.pt"
        # )
        # single_perturbation_scores = single_perturbation(test_dataset, model, device)
        # torch.save(single_perturbation_scores, f"results/scores/{seed}/single_perturbation_scores.pt")
        # print(len(single_perturbation_scores))
        # print(len(test_dataset))
        # gradient_x_input_scores = gradient_x_input(test_dataset, model, device)
        # torch.save(gradient_x_input_scores, "results/scores/gxi_scores.pt")
        # gradient_x_input_scores = torch.load("results/scores/gxi_scores.pt")

        # t1 = time.time()
        # lrp_scores = lrp(test_dataset, model, device)
        # t2 = time.time()
        # runtimes.append(t2-t1)
        # torch.save(lrp_scores, f"results/scores/{seed}/lrp_scores.pt")
        # TODO test again if time for different runtimes on balanced
        # single_perturbation_scores = torch.load(f"results/scores/{seed}/single_perturbation_scores.pt")
        t1 = time.time()
        # milli_scores = milli(test_dataset, model, n_masks=20000, initial_scores=single_perturbation_scores, device=device, alpha=0.05, beta=-1)
        
        # # shap_scores = shap(test_dataset, model, 2000, device)
        
        shap_predictor = train_shap_predictor(model, train_dataset, 2000, device)
        t2 = time.time()
        # # torch.save(milli_scores, f"results/scores/{seed}/milli_scores.pt")

        trained_shap_scores = eval_shap_predictor(shap_predictor, test_dataset, device)
        # lrp_scores = lrp(test_dataset, model, device)
        t3 = time.time()
        # torch.save(lrp_scores, f"results/scores/{seed}/lrp_scores.pt")

        torch.save(shap_predictor.state_dict(), f"models/{seed}/shap_predictor.pth")
        torch.save(trained_shap_scores, f"results/scores/{seed}/trained_shap_scores.pt")
        runtimes.append(t2-t1)
        print(t2-t1, "seconds")
        print(t3-t2, "seconds")
        
        runtimes2.append(t3-t2)
        with open("runtimes.csv", "a", encoding="utf-8") as file:
            file.write(f"{seed},{runtimes[-1]},{runtimes2[-1]}\n")
        # torch.save(milli_scores, f"results/scores/{seed}/milli_scores.pt")
        # t1 = time.time()
        # lrp_scores = lrp(test_dataset, model, device)
        # t2 = time.time()
        # runtimes.append(t2-t1)
        # torch.save(lrp, f"results/scores/{seed}/lrp_scores.pt")

        # one_removed_scores = torch.load("results/scores/one_removed_scores.pt")
        # print(
        #     "Scores are same:",
        #     all(torch.equal(a, b) for a, b in zip(new_single_scores, single_perturbation_scores))
        # )

        # milli_scores = milli(test_dataset, model, n_masks=20000, initial_scores=single_perturbation_scores, device=device, alpha=0.75, beta=1)
        # torch.save(milli_scores, "results/scores/milli_scores.pt")
        # milli_scores = torch.load("results/scores/milli_scores.pt")
        # gradient_x_input_scores = torch.load("results/scores/gxi_scores.pt")

        # integrated_gradients_scores = integrated_gradients(test_dataset, model, device, 50)
        # torch.save(integrated_gradients_scores, "results/scores/ig_scores.pt")
        # ig_scores = torch.load("results/scores/ig_scores.pt")
        # scores = {
        #     "milli": milli_scores,
        #     "attention": attention_scores,
        #     "single_perturbation": perturbation_scores["single"],
        #     "one_removed_perturbation": perturbation_scores["one_removed"],
        #     "combined_perturbation": perturbation_scores["combined"],
        #     "gradient_x_input": gradient_x_input_scores,
        #     "integrated_gradients": ig_scores,
        #     "squared_gradients": squared_gradients_scores,
        #     "lrp": lrp_scores,
        #     "rise": rise_scores,
        #     "lime": lime_scores,
        #     "shap": shap_scores,
        # }

        # shape_test(test_dataset, scores)

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

        # metrics_table = {}

        # for method, method_scores in scores.items():
        #     print(method)
        #     numerical_metrics = compute_numerical_metrics(
        #         method_scores, patch_labels, slide_ground_truth=slide_labels
        #     )
        #     plot_all_numerical_metrics(
        #         numerical_metrics, f"results/plots/numerical_metrics/{method}"
        #     )
        #     metrics_table[method] = {
        #         "AUROC": numerical_metrics["auroc"],
        #         "AP": numerical_metrics["average_precision"],
        #         "Micro-AUROC Pos": numerical_metrics["ground_truth_positive_slides"][
        #             "micro"
        #         ]["auroc"],
        #         "Micro-AP Pos": numerical_metrics["ground_truth_positive_slides"]["micro"][
        #             "ap"
        #         ],
        #         "Macro-AUROC Pos": numerical_metrics["ground_truth_positive_slides"][
        #             "macro"
        #         ]["auroc"],
        #         "Macro-AP Pos": numerical_metrics["ground_truth_positive_slides"]["macro"][
        #             "ap"
        #         ],
        #         "Pearson": numerical_metrics["normalized_correlation"],
        #         "Macro-Pearson Pos": numerical_metrics["ground_truth_positive_slides"][
        #             "macro"
        #         ]["pearson"],
        #         "Macro-Pearson Pos": numerical_metrics["ground_truth_positive_slides"][
        #             "macro"
        #         ]["pearson"],
        #     }
        # metrics_table = pd.DataFrame(metrics_table)
        # metrics_tables.append(metrics_table)
        # print(metrics_table.to_string)

        # srg_metrics = srg(attention_scores, model, device, num_bins=100)
        # print("Ascending:", srg_metrics["ascending"])
        # print("Descending:", srg_metrics["descending"])
        # print("SRG=", srg_metrics["srg"], sep="")
        # print("Time needed for computation", srg_metrics["time"])
        # print("Time needed for evaluation", evaluation_time)

        # time = compute_slides(single_perturbation_scores, "results/plots/visualized_scores/ground_truth", alpha=0.8, gt=True)
        # print(time, "needed")

        # for filename, plot_dir, description in methods:
        #     scores = torch.load(f"results/scores/{seed}/{filename}")

        #     print(plot_dir)
        #     rsrg_metrics = rsrg(
        #         model,
        #         scores,
        #         10,
        #         test_dataset,
        #         device,
        #         plot=False,
        #         method=f"{plot_dir}/{seed}",
        #         step=20,
        #     )
        #     print("R-LIF:", rsrg_metrics["R-LIF"])
        #     print("R-MIF:", rsrg_metrics["R-MIF"])
        #     print("RSRG=", rsrg_metrics["RSRG"], sep="")
        #     print("Time needed for computation", rsrg_metrics["time"])
        #     print("seed", seed)
        # print("Runtimes", runtimes)
    #     rsrg_metrics = rsrg(
    #         model,
    #         milli_scores,
    #         10,
    #         test_dataset,
    #         device,
    #         plot=True,
    #         method=f"trained_shap/{seed}",
    #         step=20,
    #     )
    #     print("R-LIF:", rsrg_metrics["R-LIF"])
    #     print("R-MIF:", rsrg_metrics["R-MIF"])
    #     print("RSRG=", rsrg_metrics["RSRG"], sep="")
    #     print("Time needed for computation", rsrg_metrics["time"])
    #     print("seed", seed)
    print(runtimes)
    print(runtimes2)
    

if __name__ == "__main__":
    main()
