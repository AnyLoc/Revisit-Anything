import sys
import torch
import logging
import multiprocessing
from datetime import datetime
from collections import OrderedDict
import wandb
import random
import numpy as np
import argparse
from vpr_model import VPRModel
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from os.path import join
from matplotlib.patches import Rectangle
import os
from glob import glob
from PIL import Image, ImageFile
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors
from os.path import join
import torchvision.transforms as T
from eval import get_val_dataset

VAL_DATASETS = [
    "MSLS",
    "MSLS_Test",
    "pitts30k_test",
    "pitts250k_test",
    "Nordland",
    "SPED",
    "pitts30k_val",
    "st_lucia",
    "tokyo247",
    "amstertime",
    "baidu",
    "sfsm",
]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset_name", type=str, help="Relative path of the dataset")
    parser.add_argument(
        "--resize",
        type=int,
        default=[322, 322],
        nargs=2,
        help="Resizing shape for images (HxW).",
    )
    parser.add_argument(
        "--baseline_paths",
        nargs="+",
        help="Paths to the baseline .npz files, e.g., ./wpca8192_last.ckpt_MSLS_predictions.npz",
        required=True,
    )
    parser.add_argument(
        "--your_method_path",
        type=str,
        help="Path to your method .npz file, e.g., ./wpca8192_last.ckpt_MSLS_predictions.npz",
        required=True,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/comparison/",
        help="name of directory on which to save the predictions, under logs/comparison",
    )

    args = parser.parse_args()
    return args


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10]


def get_recalls(args, eval_ds, predictions, loc_rad=None, ground_truth=None):
    #### For each query, check if the predictions are correct
    if ground_truth is not None:
        positives_per_query = ground_truth
    else:
        positives_per_query = eval_ds.get_positives(loc_rad)
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    if ground_truth is not None:
        recalls = recalls / eval_ds.num_queries * 100
    else:
        recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)]
    )

    return recalls, recalls_str


def extract_method_name(path):
    # Split the path and extract relevant parts
    parts = path.split("/")
    method_name = parts[-2] + "_" + parts[-1].split("_predictions.npz")[0]
    return method_name


def generate_predictions(args, eval_ds, ground_truth):
    predictions = {}
    recalls = {}
    results_list = []  # List to hold dictionaries for rows to be added to DataFrame

    # Process each baseline path, extract the baseline name, and load the predictions
    for path in args.baseline_paths:
        # Extracting the baseline name from the path
        baseline_name = extract_method_name(path)

        # Load predictions
        data = np.load(path)
        predictions[baseline_name] = data["predictions"]  # Adjust key if necessary
        recalls[baseline_name], _ = get_recalls(
            args, eval_ds, predictions[baseline_name], ground_truth=ground_truth
        )

    # Load predictions for your method
    your_method_name = "your_method"  # Or extract similarly if needed
    method_name = extract_method_name(args.your_method_path)
    data = np.load(args.your_method_path)
    predictions[your_method_name] = data["predictions"]
    # print(predictions)
    # Compute positives for the queries (assuming eval_ds is defined)
    positives_per_query = ground_truth
    recalls[method_name], _ = get_recalls(
        args, eval_ds, predictions[your_method_name], ground_truth=ground_truth
    )

    # Compare baseline predictions with your method
    for baseline, baseline_preds in predictions.items():
        if baseline != your_method_name:
            for query_index, pred in enumerate(baseline_preds):
                correct = np.any(
                    np.in1d(pred[: RECALL_VALUES[0]], positives_per_query[query_index])
                )
                if not correct:
                    # Incorrect prediction by the baseline
                    correct = np.any(
                        np.in1d(
                            predictions[your_method_name][query_index][
                                : RECALL_VALUES[0]
                            ],
                            positives_per_query[query_index],
                        )
                    )
                    if correct:
                        # Corrected by your method
                        results_list.append(
                            {
                                "Baseline": baseline,
                                "YourMethod": method_name,
                                "QueryIndex": query_index,
                                "GT": positives_per_query[query_index],
                                "BaselineTop1Prediction": pred[0],
                                "CorrectedByBaseline": False,
                                "CorrectedByYourMethod": True,
                                "YourMethodTop1Prediction": predictions[
                                    your_method_name
                                ][query_index][0],
                                "YourMethodR@1": recalls[method_name][0],
                                "BaselineR@1": recalls[baseline][0],
                            }
                        )
                else:
                    # correct prediction by the baseline
                    correct = np.any(
                        np.in1d(
                            predictions[your_method_name][query_index][
                                : RECALL_VALUES[0]
                            ],
                            positives_per_query[query_index],
                        )
                    )
                    if not correct:
                        # Not corrected by your method
                        results_list.append(
                            {
                                "Baseline": baseline,
                                "YourMethod": method_name,
                                "QueryIndex": query_index,
                                "GT": positives_per_query[query_index],
                                "BaselineTop1Prediction": pred[0],
                                "CorrectedByBaseline": True,
                                "CorrectedByYourMethod": False,
                                "YourMethodTop1Prediction": predictions[
                                    your_method_name
                                ][query_index][0],
                                "YourMethodR@1": recalls[method_name][0],
                                "BaselineR@1": recalls[baseline][0],
                            }
                        )
            #      else:
            #         #corrected by your method
            #        results_list.append({'Baseline': baseline, 'QueryIndex': query_index, 'GT': positives_per_query[query_index],'BaselineTop1Prediction': pred[0],  'CorrectedByBaseline': True, 'CorrectedByYourMethod': True,\
            #           'YourMethodTop1Prediction': predictions[your_method_name][query_index][0], 'Recall@1': recalls[baseline][0]})

    # Create directories for correct and incorrect results if they don't exist
    correct_dir = f"{args.save_dir}{args.dataset_name}/{method_name}/correct/"
    incorrect_dir = f"{args.save_dir}{args.dataset_name}/{method_name}/incorrect/"
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    # Convert list of results to DataFrame and save
    results = pd.DataFrame(results_list)
    fileName = f"{args.save_dir}{args.dataset_name}/{method_name}/{args.dataset_name}_prediction_analysis.csv"
    results.to_csv(fileName, index=False)
    print(f"Analysis saved to {fileName}")

    corrected_results = results[results["CorrectedByYourMethod"] == True]
    incorrected_results = results[results["CorrectedByYourMethod"] == False]

    def plot_and_save_images(results_df, correct):
        for index, row in results_df.iterrows():
            query_index = row["QueryIndex"]
            baseline = row["Baseline"]
            your_method = row["YourMethod"]

            # Load query image
            query_image_path = os.path.join(
                eval_ds.dataset_root, eval_ds.qImages[query_index]
            )
            query_image = Image.open(query_image_path)

            # Load the baseline's  prediction image
            baseline_pred_index = predictions[baseline][query_index][
                0
            ]  # Taking the top-1 prediction
            baseline_image_path = os.path.join(
                eval_ds.dataset_root, eval_ds.dbImages[baseline_pred_index]
            )
            baseline_image = Image.open(baseline_image_path)

            # Load your method's  prediction image
            your_pred_index = predictions[your_method_name][query_index][
                0
            ]  # Taking the top-1 prediction
            your_image_path = os.path.join(
                eval_ds.dataset_root, eval_ds.dbImages[your_pred_index]
            )
            your_image = Image.open(your_image_path)

            # Create subplot: Query vs. Baseline  vs. Your Method /
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Query image subplot
            ax[0].imshow(query_image)
            ax[0].set_title(f"Query Image (Index: {query_index})", fontsize=7)
            ax[0].axis("off")

            # Baseline  image subplot
            ax[1].imshow(baseline_image)
            ax[1].set_title(f"{baseline} (Index: {baseline_pred_index})", fontsize=7)
            rect_color = "r" if correct else "g"
            ax[1].add_patch(
                Rectangle(
                    (0, 0),
                    baseline_image.width,
                    baseline_image.height,
                    linewidth=1,
                    edgecolor=rect_color,
                    facecolor="none",
                )
            )
            ax[1].axis("off")

            # Your method's prediction image subplot
            ax[2].imshow(your_image)
            ax[2].set_title(f"{your_method} (Index: {your_pred_index})", fontsize=7)
            rect_color = "g" if correct else "r"
            ax[2].add_patch(
                Rectangle(
                    (0, 0),
                    your_image.width,
                    your_image.height,
                    linewidth=1,
                    edgecolor=rect_color,
                    facecolor="none",
                )
            )
            ax[2].axis("off")

            plt.tight_layout()
            plt.show()

            # Save the figure in the appropriate folder
            save_dir = correct_dir if correct else incorrect_dir
            fig.savefig(
                os.path.join(
                    save_dir,
                    f"{args.dataset_name}_{baseline}_{'correct' if correct else 'incorrect'}_{query_index}.png",
                )
            )
            plt.close(fig)

    # Plot and save correct results
    plot_and_save_images(corrected_results, correct=True)
    # Plot and save incorrect results
    plot_and_save_images(incorrected_results, correct=False)


def main():
    args = parse_args()

    if args.dataset_name in VAL_DATASETS:
        ds, num_references, num_queries, ground_truth = get_val_dataset(
            args.dataset_name, args.resize
        )
        generate_predictions(args, ds, ground_truth)
        print(args.dataset_name)


if __name__ == "__main__":
    main()
