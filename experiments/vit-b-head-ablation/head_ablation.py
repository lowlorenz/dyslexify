"""
Head Ablation Experiment Module

This module implements a head ablation experiment for vision transformer models.
The experiment systematically ablates each attention head individually and measures
the impact on model performance for both normal and typographic images.

The main class `HeadAblationExperiment` provides functionality to:
- Load and configure vision transformer models
- Set up ImageNet-100 dataset with typographic perturbations
- Calculate text features for zero-shot classification
- Run head ablation experiments by ablating each attention head individually
- Track and report performance metrics for each head

Author: [Your Name]
Date: [Date]
"""

import argparse
import json
import os
from typing import Any, Callable, List, Tuple, Dict

import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dyslexify.cache.collector import change_attn_implementation_to_hookable
from dyslexify.config import MODELS
from dyslexify.dataset.ImageNet100 import ImageNet100
from dyslexify.zeroshot import calculate_text_features


def create_zero_cls_attention_result_hook(head_idx: int) -> Callable:
    """
    Create a hook function that zeros out the attention output for a specific head.

    This hook is used to ablate (disable) specific attention heads during experiments
    by setting their output to zero.

    Args:
        head_idx (int): Index of the attention head to zero out

    Returns:
        Callable: Hook function that takes attention output, weights, and key/value tensors
                 and returns modified attention output with the specified head zeroed
    """

    def zero_cls_attention_result_hook(attn_output, attn_weights, q, k, v):
        attn_output[:, head_idx, 0, :] = 0
        return attn_output

    return zero_cls_attention_result_hook


def create_zero_spatial_attention_result_hook(head_idx: int) -> Callable:
    """
    Create a hook function that zeros out the attention output for a specific head.

    This hook is used to ablate (disable) specific attention heads during experiments
    by setting their output to zero.

    Args:
        head_idx (int): Index of the attention head to zero out

    Returns:
        Callable: Hook function that takes attention output, weights, and key/value tensors
                 and returns modified attention output with the specified head zeroed
    """

    def zero_cls_attention_result_hook(attn_output, attn_weights, q, k, v):
        attn_output[:, head_idx, 1:, :] = 0
        return attn_output

    return zero_cls_attention_result_hook


class DatasetSubset:
    """
    A custom subset class that preserves the original dataset's attributes
    needed for text feature generation.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        # Preserve important attributes from the original dataset
        for attr in ["classes", "templates", "class_to_idx"]:
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))
            else:
                print(
                    f"DatasetSubset: Warning - original dataset missing {attr} attribute"
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the head ablation experiment.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Run head ablation experiment on vision transformer attention heads"
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
        default=128,
        help="Batch size for data loading (default: 128)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of workers for data loading (default: 32)",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/datasets/imagenet-100-typo",
        help="Root directory for the ImageNet-100 dataset with typographic perturbations (default: /datasets/imagenet-100-typo)",
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=2500,
        help="Number of samples to use from the dataset (default: 2500)",
    )

    parser.add_argument(
        "--subset-seed",
        type=int,
        default=42,
        help="Random seed for reproducible subset sampling (default: 42)",
    )

    # Head ablation configuration
    parser.add_argument(
        "--head-type",
        type=str,
        default="cls",
        choices=["cls", "spatial"],
        help="Type of attention heads to ablate: 'cls' for CLS token heads or 'spatial' for spatial token heads (default: cls)",
    )

    # Output configuration
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/experiments/vit-b-head-ablation",
        help="Directory to save experiment results (default: results/experiments/vit-b-head-ablation)",
    )

    # Verbosity
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output during experiment execution",
    )

    return parser.parse_args()


class HeadAblationExperiment:
    """
        A class to conduct head ablation experiments on vision transformer attention heads.
    del
        This experiment systematically ablates each attention head individually and measures
        the impact on model performance for both normal and typographic images. It provides
        detailed analysis of which heads are most important for different types of inputs.

        Attributes:
            model_short_name (str): Short name identifier for the model
            model_name (str): Full model name from configuration
            pretrained (str): Pretrained model variant
            device (str): Device to run experiments on (e.g., 'cuda:0')
            batch_size (int): Batch size for data loading
            num_workers (int): Number of workers for data loading
            dataset_root (str): Root directory for the dataset
            subset_size (int): Number of samples to use from the dataset
            subset_seed (int): Random seed for subset sampling
            head_type (str): Type of attention heads to ablate ('cls' or 'spatial')
            save_dir (str): Directory to save experiment results
            verbose (bool): Whether to enable verbose output
            model: The loaded vision transformer model
            dataset: ImageNet-100 dataset with typographic perturbations
            dataloader: DataLoader for the dataset
            text_features (torch.Tensor): Pre-computed text features for zero-shot classification
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads per layer
    """

    def __init__(
        self,
        model_short_name: str,
        device: str,
        batch_size: int,
        num_workers: int,
        dataset_root: str = "/datasets/imagenet-100-typo",
        subset_size: int = 2500,
        subset_seed: int = 42,
        head_type: str = "cls",
        save_dir: str = "results/experiments/vit-b-head-ablation",
        verbose: bool = False,
    ):
        """
        Initialize the HeadAblationExperiment.

        Args:
            model_short_name (str): Short name identifier for the model (e.g., 'vit-b')
            device (str): Device to run experiments on (e.g., 'cuda:0')
            batch_size (int): Batch size for data loading
            num_workers (int): Number of workers for data loading
            dataset_root (str, optional): Root directory for the dataset. Defaults to "/datasets/imagenet-100-typo".
            subset_size (int, optional): Number of samples to use from the dataset. Defaults to 2500.
            subset_seed (int, optional): Random seed for subset sampling. Defaults to 42.
            head_type (str, optional): Type of attention heads to ablate: 'cls' for CLS token heads or 'spatial' for spatial token heads. Defaults to "cls".
            save_dir (str, optional): Directory to save experiment results. Defaults to "results/experiments/vit-b-head-ablation".
            verbose (bool, optional): Whether to enable verbose output. Defaults to False.
        """
        self.model_short_name = model_short_name
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_root = dataset_root
        self.subset_size = subset_size
        self.subset_seed = subset_seed
        self.head_type = head_type
        self.save_dir = save_dir
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing experiment with model: {self.model_short_name}")
            print(
                f"Device: {self.device}, Batch size: {self.batch_size}, Workers: {self.num_workers}"
            )
            print(f"Subset size: {self.subset_size}, Subset seed: {self.subset_seed}")
            print(f"Head type: {self.head_type}")

        self.model, transform, tokenizer = self.load_model()
        self.dataset, self.dataloader = self.setup_dataset(transform)
        self.text_features = calculate_text_features(
            self.model, self.dataset, tokenizer, self.device
        )

        # Get model dimensions
        self.num_layers = len(self.model.visual.transformer.resblocks)
        self.num_heads = self.model.visual.transformer.resblocks[0].attn.num_heads

        torch.set_float32_matmul_precision("high")
        change_attn_implementation_to_hookable(self.model)
        self.model = torch.compile(self.model)

    def _create_subset(self, dataset, subset_size: int, seed: int = 42):
        """Create a subset of the dataset."""
        if subset_size >= len(dataset):
            print(
                f"Warning: subset_size ({subset_size}) >= dataset size ({len(dataset)}). Using full dataset."
            )
            return dataset

        # Set random seed for reproducible sampling
        import random

        random.seed(seed)

        # Create random indices
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]

        # Create subset using our custom class
        subset = DatasetSubset(dataset, subset_indices)

        print(
            f"Created subset with {len(subset)} samples from {len(dataset)} total samples"
        )

        # Debug: Check if important attributes are preserved
        if hasattr(subset, "classes"):
            print(f"Subset has {len(subset.classes)} classes preserved")
        else:
            print("Warning: Subset does not have 'classes' attribute")

        return subset

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

        dataset = ImageNet100(
            root=self.dataset_root,
            split="train",
            preprocess=transform,
            position="random",
        )

        # Create subset
        dataset = self._create_subset(dataset, self.subset_size, self.subset_seed)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        if self.verbose:
            print(f"Dataset loaded with {len(dataset)} samples")
            print(f"Number of classes: {len(dataset.classes)}")

        return dataset, dataloader

    def load_model(self) -> Tuple[Any, Any, Any]:
        """
        Load the vision transformer model and associated components.

        Returns:
            Tuple[Any, Any, Any]: Model, transform, and tokenizer
        """
        if self.verbose:
            print(
                f"Loading model: {self.model_name} with pretrained: {self.pretrained}"
            )

        model, _, transform = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )
        model.to(self.device)

        tokenizer = open_clip.get_tokenizer(self.model_name)

        if self.verbose:
            print("Model loaded successfully")

        return model, transform, tokenizer

    def run_experiment(self) -> None:
        """
        Run the complete head ablation experiment.

        This is the main entry point for running the experiment. It evaluates
        the impact of ablating each attention head individually.
        """
        if self.verbose:
            print("Starting head ablation experiment...")

        self.run_head_ablation()

    def run_head_ablation(self) -> None:
        """
        Execute the head ablation experiment using an optimized loop approach.

        This method evaluates the impact of ablating each attention head individually
        by iterating through the dataloader once and for each batch, evaluating all
        head ablations. This is more efficient as it processes all heads for each
        batch of data rather than processing all data for each head.

        The experiment:
        1. Computes baseline accuracy for each batch
        2. For each batch, evaluates all head ablations
        3. Measures accuracy changes for both normal and typographic images
        4. Saves detailed results in JSON format
        """
        if self.verbose:
            print(f"Starting head ablation experiment with optimized loop approach...")
            print(f"Ablating {self.head_type} attention heads")

        results = []
        blocks = self.model.visual.transformer.resblocks
        total_heads = self.num_layers * self.num_heads

        # Initialize counters for accuracy calculation
        baseline_correct_normal = 0
        baseline_correct_typo = 0
        baseline_total = 0

        # Initialize counters for each head ablation
        head_correct_normal = {i: 0 for i in range(total_heads)}
        head_correct_typo = {i: 0 for i in range(total_heads)}
        head_total = {i: 0 for i in range(total_heads)}

        # Outer loop: iterate through dataloader
        with tqdm(self.dataloader, desc="Processing batches") as batch_pbar:
            for batch_idx, (
                normal_images,
                typo_images,
                normal_labels,
                typo_labels,
            ) in enumerate(batch_pbar):
                normal_images = normal_images.to(self.device)
                typo_images = typo_images.to(self.device)
                normal_labels = normal_labels.to(self.device)
                typo_labels = typo_labels.to(self.device)

                # Inner loop: evaluate baseline and all head ablations for this batch
                with tqdm(
                    total=1 + total_heads, desc=f"Batch {batch_idx}", leave=False
                ) as head_pbar:
                    # Baseline evaluation (no hooks)
                    with torch.no_grad():
                        # Forward pass for normal images
                        normal_features = self.model.encode_image(normal_images)
                        normal_features = normal_features / normal_features.norm(
                            dim=-1, keepdim=True
                        )
                        normal_logits = 100 * normal_features @ self.text_features.T
                        normal_predictions = normal_logits.argmax(dim=-1)
                        normal_correct = (normal_predictions == normal_labels).float()

                        # Forward pass for typographic images
                        typo_features = self.model.encode_image(typo_images)
                        typo_features = typo_features / typo_features.norm(
                            dim=-1, keepdim=True
                        )
                        typo_logits = 100 * typo_features @ self.text_features.T
                        typo_predictions = typo_logits.argmax(dim=-1)
                        typo_correct = (typo_predictions == typo_labels).float()

                        # Update baseline counters
                        baseline_correct_normal += normal_correct.sum().item()
                        baseline_correct_typo += typo_correct.sum().item()
                        baseline_total += len(normal_labels) + len(typo_labels)

                    head_pbar.update(1)

                    # Evaluate each head ablation
                    for head_idx in range(total_heads):
                        layer = head_idx // self.num_heads
                        head = head_idx % self.num_heads

                        # Create and register hook for this specific head
                        if self.head_type == "cls":
                            hook = create_zero_cls_attention_result_hook(head)
                        else:
                            hook = create_zero_spatial_attention_result_hook(head)

                        blocks[layer].attn.register_attn_result_hook(hook)

                        # Forward pass for normal images with ablation
                        with torch.no_grad():
                            normal_features = self.model.encode_image(normal_images)
                            normal_features = normal_features / normal_features.norm(
                                dim=-1, keepdim=True
                            )
                            normal_logits = 100 * normal_features @ self.text_features.T
                            normal_predictions = normal_logits.argmax(dim=-1)
                            normal_correct = (
                                normal_predictions == normal_labels
                            ).float()

                            # Forward pass for typographic images with ablation
                            typo_features = self.model.encode_image(typo_images)
                            typo_features = typo_features / typo_features.norm(
                                dim=-1, keepdim=True
                            )
                            typo_logits = 100 * typo_features @ self.text_features.T
                            typo_predictions = typo_logits.argmax(dim=-1)
                            typo_correct = (typo_predictions == typo_labels).float()

                            # Update head counters
                            head_correct_normal[head_idx] += normal_correct.sum().item()
                            head_correct_typo[head_idx] += typo_correct.sum().item()
                            head_total[head_idx] += len(normal_labels) + len(
                                typo_labels
                            )

                        # Remove the hook to restore the model to baseline
                        if hasattr(blocks[layer].attn, "remove_attention_pattern_hook"):
                            blocks[layer].attn.remove_attention_pattern_hook()

                        head_pbar.update(1)

                # Update progress bar
                batch_pbar.set_postfix(
                    {
                        "Baseline Normal": f"{baseline_correct_normal/max(baseline_total//2, 1):.4f}",
                        "Baseline Typo": f"{baseline_correct_typo/max(baseline_total//2, 1):.4f}",
                    }
                )

        # Calculate final accuracies and results
        baseline_normal_acc = baseline_correct_normal / max(baseline_total // 2, 1)
        baseline_typo_acc = baseline_correct_typo / max(baseline_total // 2, 1)

        if self.verbose:
            print(f"Baseline accuracy: {baseline_normal_acc:.4f}")
            print(f"Baseline typographic accuracy: {baseline_typo_acc:.4f}")

        # Process results for each head
        for head_idx in range(total_heads):
            layer = head_idx // self.num_heads
            head = head_idx % self.num_heads

            ablated_normal_acc = head_correct_normal[head_idx] / max(
                head_total[head_idx] // 2, 1
            )
            ablated_typo_acc = head_correct_typo[head_idx] / max(
                head_total[head_idx] // 2, 1
            )

            # Calculate deltas
            normal_delta = ablated_normal_acc - baseline_normal_acc
            typo_delta = ablated_typo_acc - baseline_typo_acc

            # Store results
            head_result = {
                "layer": layer,
                "head": head,
                "head_index": head_idx,
                "baseline_normal_acc": baseline_normal_acc,
                "baseline_typo_acc": baseline_typo_acc,
                "ablated_normal_acc": ablated_normal_acc,
                "ablated_typo_acc": ablated_typo_acc,
                "normal_delta": normal_delta,
                "typo_delta": typo_delta,
                "total_delta": normal_delta + typo_delta,
            }

            results.append(head_result)

            if self.verbose:
                print(f"Head {head_idx} (Layer {layer}, Head {head}):")
                print(f"  Normal delta: {normal_delta:.4f}")
                print(f"  Typo delta: {typo_delta:.4f}")
                print(f"  Total delta: {head_result['total_delta']:.4f}")

        # Sort results by total impact (most impactful heads first)
        results.sort(key=lambda x: abs(x["total_delta"]), reverse=True)

        # Save results
        save_path = os.path.join(self.save_dir, self.model_short_name)
        os.makedirs(save_path, exist_ok=True)

        if self.verbose:
            print(f"Saving results to {save_path}")

        # Save detailed results with head type in filename
        results_filename = f"head_ablation_results_{self.head_type}.json"
        with open(os.path.join(save_path, results_filename), "w") as f:
            json.dump(results, f, indent=2)

        # Create summary statistics
        summary = {
            "model": self.model_short_name,
            "head_type": self.head_type,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "total_heads": total_heads,
            "subset_size": self.subset_size,
            "subset_seed": self.subset_seed,
            "baseline_normal_acc": baseline_normal_acc,
            "baseline_typo_acc": baseline_typo_acc,
            "top_10_most_impactful_heads": results[:10],
            "top_10_least_impactful_heads": results[-10:],
            "statistics": {
                "mean_normal_delta": sum(r["normal_delta"] for r in results)
                / len(results),
                "mean_typo_delta": sum(r["typo_delta"] for r in results) / len(results),
                "std_normal_delta": torch.std(
                    torch.tensor([r["normal_delta"] for r in results])
                ).item(),
                "std_typo_delta": torch.std(
                    torch.tensor([r["typo_delta"] for r in results])
                ).item(),
                "max_normal_delta": max(r["normal_delta"] for r in results),
                "min_normal_delta": min(r["normal_delta"] for r in results),
                "max_typo_delta": max(r["typo_delta"] for r in results),
                "min_typo_delta": min(r["typo_delta"] for r in results),
            },
        }

        summary_filename = f"head_ablation_summary_{self.head_type}.json"
        with open(os.path.join(save_path, summary_filename), "w") as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print("Experiment completed successfully!")
            print(f"Results saved to {save_path}")
            print(f"Top 3 most impactful heads:")
            for i, result in enumerate(results[:3]):
                print(
                    f"  {i+1}. Layer {result['layer']}, Head {result['head']}: "
                    f"Normal Δ={result['normal_delta']:.4f}, "
                    f"Typo Δ={result['typo_delta']:.4f}"
                )


def main():
    """
    Main function to run the head ablation experiment with command line arguments.
    """
    args = parse_arguments()

    print("=" * 60)
    print("HEAD ABLATION EXPERIMENT")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.num_workers}")
    print(f"Subset size: {args.subset_size}")
    print(f"Subset seed: {args.subset_seed}")
    print(f"Head type: {args.head_type}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Save directory: {args.save_dir}")
    print(f"Verbose: {args.verbose}")
    print("=" * 60)

    experiment = HeadAblationExperiment(
        model_short_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset_root=args.dataset_root,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        head_type=args.head_type,
        save_dir=args.save_dir,
        verbose=args.verbose,
    )
    experiment.run_experiment()


if __name__ == "__main__":
    main()
