"""
Greedy Selection Experiment Module

This module implements a greedy selection algorithm for identifying the most important
attention heads in vision transformer models. The experiment systematically ablates
attention heads based on their typographic scores and measures the impact on model
performance.

The main class `GreedySelectionExperiment` provides functionality to:
- Load and configure vision transformer models
- Set up ImageNet-100 dataset with typographic perturbations
- Calculate text features for zero-shot classification
- Run greedy selection experiments by iteratively ablating attention heads
- Track and report performance metrics
"""

import argparse
import json
import os
import random
from typing import Any, Callable, List, Tuple, Dict

import einops
import open_clip
import torch
from torch.utils.data import DataLoader, Subset

from dyslexify.cache.collector import change_attn_implementation_to_hookable
from dyslexify.cache.hooks import (
    create_zero_cls_attention_result_hook,
    create_zero_spatial_attention_result_hook,
)
from dyslexify.config import MODELS
from dyslexify.dataset import Melanoma, ChestXRay, ImageNet100
from dyslexify.zeroshot import zeroshot_classifier, calculate_text_features
from pathlib import Path


class SmartSubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.classes = dataset.classes
        self.templates = dataset.templates

    def __repr__(self):
        return f"SmartSubset(dataset={self.dataset})"


# def create_zero_spatial_attention_result_hook(head_idx: int) -> Callable:


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the greedy selection experiment.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run greedy selection experiment on vision transformer attention heads"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="vit-b",
        choices=list(MODELS.keys()),
        help="Model short name to use for the experiment (default: vit-b)",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the experiment on (default: cuda:0)",
    )

    # Data loading configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for data loading (default: 512)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)",
    )

    # Experiment configuration
    parser.add_argument(
        "--stop-at-delta",
        type=float,
        default=0.01,
        help="Threshold for stopping experiment when accuracy degradation is below this value (default: 0.01)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/datasets/imagenet-100-typo",
        help="Root directory for the ImageNet-100 dataset with typographic perturbations (default: /datasets/imagenet-100-typo)",
    )

    # Output configuration
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/experiments/greedy_selection",
        help="Directory to save experiment results (default: results/experiments/greedy_selection)",
    )

    # Greedy selection mode
    parser.add_argument(
        "--mode",
        type=str,
        default="cls",
        choices=["cls", "spatial"],
        help="Mode to run the experiment in (default: cls)",
    )

    # Dataset size
    parser.add_argument(
        "--dataset-size",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use (default: 1.0)",
    )

    return parser.parse_args()


class GreedySelectionExperiment:
    """
    A class to conduct greedy selection experiments on vision transformer attention heads.

    This experiment systematically ablates attention heads based on their typographic
    scores to identify the most important heads for model performance. It tracks
    accuracy changes and provides detailed reporting of results.

    Attributes:
        model_short_name (str): Short name identifier for the model
        model_name (str): Full model name from configuration
        pretrained (str): Pretrained model variant
        device (str): Device to run experiments on (e.g., 'cuda:0')
        batch_size (int): Batch size for data loading
        num_workers (int): Number of workers for data loading
        stop_at_delta (float): Threshold for stopping the experiment when accuracy
                              degradation is below this value
        dataset_root (str): Root directory for the dataset
        model: The loaded vision transformer model
        dataset: ImageNet-100 dataset with typographic perturbations
        dataloader: DataLoader for the dataset
        text_features (torch.Tensor): Pre-computed text features for zero-shot classification
        T (torch.Tensor): Typographic scores matrix
        sorted_scores (List[Tuple[float, int, int]]): Sorted list of (score, layer, head) tuples
    """

    def __init__(
        self,
        model_short_name: str,
        device: str,
        batch_size: int,
        num_workers: int,
        stop_at_delta: float = 0.01,
        dataset_root: str = "/datasets/imagenet-100-typo",
        mode: str = "cls",
        dataset_size: float = 1.0,
    ):
        """
        Initialize the GreedySelectionExperiment.

        Args:
            model_short_name (str): Short name identifier for the model (e.g., 'vit-big-g')
            device (str): Device to run experiments on (e.g., 'cuda:0')
            batch_size (int): Batch size for data loading
            num_workers (int): Number of workers for data loading
            stop_at_delta (float, optional): Threshold for stopping experiment when accuracy
                                           degradation is below this value. Defaults to 0.01.
            dataset_root (str, optional): Root directory for the dataset. Defaults to "/datasets/imagenet-100-typo".

        """
        self.model_short_name = model_short_name
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stop_at_delta = stop_at_delta
        self.dataset_root = dataset_root
        self.dataset_size = dataset_size
        self.save_dir = Path(
            f"results/experiments/greedy_selection/{self.model_short_name}"
        )

        if mode not in ["cls", "spatial"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
        self.get_hook_fn = (
            create_zero_cls_attention_result_hook
            if self.mode == "cls"
            else create_zero_spatial_attention_result_hook
        )
        self.T, self.sorted_scores = self.load_typographic_scores()

        self.model, transform, tokenizer = self.load_model()
        self.dataset, self.dataloader = self.setup_dataset(transform)
        self.text_features = calculate_text_features(
            self.model, self.dataset, tokenizer, self.device
        )

        torch.set_float32_matmul_precision("high")
        self.model = torch.compile(self.model)
        change_attn_implementation_to_hookable(self.model)

    def setup_dataset(self, transform) -> Tuple[ImageNet100, DataLoader]:
        """
        Set up the ImageNet-100 dataset with typographic perturbations.

        Args:
            transform: Image transformation pipeline. If None, creates default transform
                      for the model.

        Returns:
            Tuple[ImageNet100, DataLoader]: Dataset and DataLoader for the experiment
        """
        if transform is None:
            _, _, transform = open_clip.create_model_and_transforms(
                self.model_name, self.pretrained
            )

        if self.model_short_name == "whylesionclip":
            dataset = Melanoma(
                root="/datasets/melanoma_cancer_dataset_typo",
                split="train",
                preprocess=transform,
                position="random",
            )
        else:
            dataset = ImageNet100(
                root=self.dataset_root,
                split="train",
                preprocess=transform,
                position="random",
            )

        if self.dataset_size < 1.0:
            dataset = SmartSubset(
                dataset,
                random.sample(
                    range(len(dataset)), int(len(dataset) * self.dataset_size)
                ),
            )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return dataset, dataloader

    def load_typographic_scores(
        self,
    ) -> Tuple[torch.Tensor, List[Tuple[float, int, int]]]:
        """
        Load pre-computed typographic scores for all attention heads.

        The typographic scores indicate how important each attention head is
        for handling typographic perturbations. Higher scores indicate more
        important heads.

        Returns:
            Tuple[torch.Tensor, List[Tuple[float, int, int]]]:
                - T: Matrix of typographic scores (layers x heads)
                - sorted_scores: List of (score, layer, head) tuples sorted by score
                                in descending order
        """
        scores_path = f"results/experiments/typographic_scores/{self.model_short_name}/typographic_scores_{self.mode}.pt"

        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Typographic scores not found at {scores_path}")

        T = torch.load(scores_path)

        # sort the typographic scores by descending order
        layer, head = T.shape
        sorted_scores = [
            (T[i, j].item(), i, j) for i in range(layer) for j in range(head)
        ]
        sorted_scores.sort(key=lambda x: x[0], reverse=True)

        return T, sorted_scores

    def load_model(self) -> Tuple[Any, Any, Any]:
        """
        Load the vision transformer model and associated components.

        Returns:
            Tuple[Any, Any, Any]: Model, transform, and tokenizer
        """
        model, _, transform = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )
        model.to(self.device)

        tokenizer = open_clip.get_tokenizer(self.model_name)

        return model, transform, tokenizer

    def load_or_run_experiment(self) -> List[Dict]:
        """
        Load the results of the greedy selection experiment.
        """
        heads_path = self.save_dir / f"ablated_heads_{self.mode}.json"
        results_path = self.save_dir / f"results_{self.mode}.json"

        if heads_path.exists() and results_path.exists():
            with open(heads_path, "r") as f:
                heads = json.load(f)
            with open(results_path, "r") as f:
                results = json.load(f)
            return results, heads

        return self.run_experiment()

    def run_experiment(self) -> None:
        """
        Run the complete greedy selection experiment.

        This is the main entry point for running the experiment. It calls
        the greedy selection algorithm to identify important attention heads.
        """
        return self.run_greedy_selection()

    def run_greedy_selection(self) -> None:
        """
        Execute the greedy selection algorithm for attention heads.

        This method iteratively ablates attention heads based on their typographic
        scores, starting with the highest-scoring heads. It measures the impact
        on model accuracy and stops when the accuracy degradation falls below
        the specified threshold.

        The algorithm:
        1. Computes baseline accuracy
        2. Iteratively ablates heads in order of typographic scores
        3. Measures accuracy changes after each ablation
        4. Stops when accuracy degradation is below threshold
        5. Reports detailed results in a transposed table format
        """
        baseline_acc, baseline_typo_acc = zeroshot_classifier(
            self.model, self.dataloader, self.text_features, self.device
        )

        current_acc = baseline_acc
        current_typo_acc = baseline_typo_acc

        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print(f"Baseline typographic accuracy: {baseline_typo_acc:.4f}")

        ablated_heads = []
        results_data = []  # Store all results for transposed table

        blocks = self.model.visual.transformer.resblocks

        eps = 0.001
        under_eps_counter = 0
        stop_after_n_times_under_eps = 10

        for score, layer, head in self.sorted_scores:
            if self.mode == "spatial" and layer == len(blocks) - 1:
                print(
                    f"Skipping layer {layer} in mode {self.mode} which does not influence the output"
                )
                continue

            # Remove all hooks from all blocks
            for block in blocks:
                block.attn.remove_all_hooks()

            # Register the hooks for the ablated heads
            for _layer, _head in ablated_heads:
                hook = self.get_hook_fn(_head)
                blocks[_layer].attn.register_attn_result_hook(hook)

            # Register the hook for the current head
            hook = self.get_hook_fn(head)
            blocks[layer].attn.register_attn_result_hook(hook)

            ablated_acc, ablated_typo_acc = zeroshot_classifier(
                self.model, self.dataloader, self.text_features, self.device
            )

            # Determine whether to keep this head based on current_typo_delta
            curr_delta = ablated_acc - current_acc
            curr_typo_delta = ablated_typo_acc - current_typo_acc
            skipped_flag = curr_typo_delta < eps

            # Store results for transposed table
            results_data.append(
                {
                    "layer": layer,
                    "head": head,
                    "score": score,
                    "ablated_acc": ablated_acc,
                    "baseline_acc": baseline_acc,
                    "delta": ablated_acc - baseline_acc,
                    "curr_delta": curr_delta,
                    "ablated_typo_acc": ablated_typo_acc,
                    "baseline_typo_acc": baseline_typo_acc,
                    "typo_delta": ablated_typo_acc - baseline_typo_acc,
                    "curr_typo_delta": curr_typo_delta,
                    "skipped": skipped_flag,
                }
            )

            self.log_results(results_data)

            if under_eps_counter > stop_after_n_times_under_eps:
                print(f"Stopping at iteration {len(results_data)}")
                break

            # If this head is skipped, do not update current metrics
            if skipped_flag:
                under_eps_counter += 1
                continue

            under_eps_counter = 0

            current_acc = ablated_acc
            current_typo_acc = ablated_typo_acc
            ablated_heads.append((layer, head))

            # Early stop check based on baseline degradation
            if baseline_acc - ablated_acc > self.stop_at_delta:
                print(f"Stopping at iteration {len(results_data)}")
                break

        # Save results
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.save_dir / f"results_{self.mode}.json", "w") as f:
            json.dump(results_data, f, indent=2)

        with open(self.save_dir / f"ablated_heads_{self.mode}.json", "w") as f:
            json.dump(ablated_heads, f, indent=2)

        return results_data, ablated_heads

    def plot_results(self, results_data: List[Dict]) -> None:
        """
        Plot the results of the greedy selection experiment with reference styling.
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        from experiments.plot_config import COLOR_NORMAL, COLOR_TYPO, FIGSIZE

        # Filter out skipped results and prepare data
        results_df = pd.DataFrame(results_data)
        non_skipped_df = results_df[results_df.skipped == False].reset_index(drop=True)

        # Prepare x-axis data (number of ablated heads)
        steps = np.arange(0, len(non_skipped_df) + 1)  # +1 to include baseline

        # Prepare accuracy data including baseline
        baseline_acc = results_data[0]["baseline_acc"]
        baseline_typo_acc = results_data[0]["baseline_typo_acc"]

        regular_accuracies = [baseline_acc] + non_skipped_df["ablated_acc"].tolist()
        typo_accuracies = [baseline_typo_acc] + non_skipped_df[
            "ablated_typo_acc"
        ].tolist()

        # Create figure with reference styling
        fig, ax = plt.subplots(figsize=(5, 3))

        # Plot with reference colors and styling
        ax.plot(
            steps,
            [acc * 100 for acc in regular_accuracies],  # Convert to percentage
            color=COLOR_NORMAL,
            marker="o",
            markersize=4,
            label="ImageNet-100",
            linestyle="-",
        )
        ax.plot(
            steps,
            [acc * 100 for acc in typo_accuracies],  # Convert to percentage
            color=COLOR_TYPO,
            marker="o",
            markersize=4,
            label="ImageNet-100-Typo",
            linestyle="--",
        )

        # Customize plot appearance
        ax.set_xlabel("Number of Ablated Heads")
        ax.set_ylabel("Zeroshot Accuracy (%)")
        ax.set_title(f"Greedy Selection - {self.model_short_name.upper()}")

        # Remove plot edges (spines)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.set_ylim(45, 90)

        # Add grid and legend
        ax.grid(True, linestyle="-", alpha=0.3)
        ax.legend(loc="lower right")

        # Set x-axis limits and ticks
        ax.set_xlim(-0.5, len(steps) - 0.5)

        # Set x-axis ticks based on data length
        if len(steps) <= 10:
            ax.set_xticks(steps)
        else:
            tick_positions = np.arange(0, len(steps), 5)
            ax.set_xticks(tick_positions)

        plt.tight_layout()

        # Save plot
        save_path = self.save_dir / f"ablation_plot_{self.mode}.png"
        plt.savefig(save_path, format="png", bbox_inches="tight", dpi=150)
        save_path = self.save_dir / f"ablation_plot_{self.mode}.svg"
        plt.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

        return fig, ax

    def log_results(self, results_data: List[Dict]) -> None:
        """
        Log the results of the greedy selection experiment.
        """
        # Print transposed table after each step
        print("\n" + "=" * 120)
        print(f"TRANSPOSED RESULTS TABLE - AFTER ITERATION {len(results_data)}")
        print("=" * 120)

        # Print header row with iteration numbers
        header = f"{'Metric':<20}"
        for i, data in enumerate(results_data):
            header += f" {'Iter '+str(i+1):<10}"
        print(header)
        print("-" * 120)

        # Print each metric as a row
        metrics = [
            ("Layer", "layer"),
            ("Head", "head"),
            ("Score", "score"),
            ("Ablated Acc", "ablated_acc"),
            ("Baseline Acc", "baseline_acc"),
            ("Delta", "delta"),
            ("Curr Delta", "curr_delta"),
            ("Ablated Typo", "ablated_typo_acc"),
            ("Baseline Typo", "baseline_typo_acc"),
            ("Typo Delta", "typo_delta"),
            ("Curr Typo Delta", "curr_typo_delta"),
        ]

        for metric_name, metric_key in metrics:
            row = f"{metric_name:<20}"
            for data in results_data:
                value = data[metric_key]
                if isinstance(value, float):
                    row += f" {value:<10.3f}"
                else:
                    row += f" {value:<10}"
            print(row)
        # Print Skipped row with ✓ for skipped heads and ✗ otherwise
        skipped_row = f"{'Skipped':<20}"
        for data in results_data:
            skipped_row += f" {'✓' if data.get('skipped') else '✗':<10}"
        print(skipped_row)


def main():
    """
    Main function to run the greedy selection experiment with command line arguments.
    """
    args = parse_arguments()

    print("=" * 60)
    print("GREEDY SELECTION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Stop threshold: {args.stop_at_delta}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Mode: {args.mode}")
    print(f"Dataset size: {args.dataset_size}")
    print("=" * 60)

    experiment = GreedySelectionExperiment(
        model_short_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stop_at_delta=args.stop_at_delta,
        dataset_root=args.dataset_root,
        dataset_size=args.dataset_size,
        mode=args.mode,
    )
    experiment.run_experiment()


if __name__ == "__main__":
    main()
