import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from experiments.plot_config import COLOR_NORMAL, COLOR_TYPO, FIGSIZE
from typing import List, Dict
from pathlib import Path
import json

model_name_dict = {
    "vit-b": "ViT-B",
    "vit-l": "ViT-L",
    "vit-h": "ViT-H",
    "vit-g": "ViT-G",
    "vit-big-g": "ViT-BigG",
    "whylesionclip": "WhylesionCLIP",
}


def plot_ablation_curve(
    results_data: List[Dict], model_short_name: str, mode: str, save_dir: Path
) -> None:
    """
    Plot the results of the greedy selection experiment with reference styling.
    """

    # Filter out skipped results and prepare data
    results_df = pd.DataFrame(results_data)
    non_skipped_df = results_df[results_df.skipped == False].reset_index(drop=True)

    # Prepare x-axis data (number of ablated heads)
    steps = np.arange(0, len(non_skipped_df) + 1)  # +1 to include baseline

    # Prepare accuracy data including baseline
    baseline_acc = results_data[0]["baseline_acc"]
    baseline_typo_acc = results_data[0]["baseline_typo_acc"]

    regular_accuracies = [baseline_acc] + non_skipped_df["ablated_acc"].tolist()
    typo_accuracies = [baseline_typo_acc] + non_skipped_df["ablated_typo_acc"].tolist()

    # Set labels based on model
    if model_short_name == "whylesionclip":
        normal_label = "ISIC2019"
        typo_label = "ISIC2019-Typo"
    else:
        normal_label = "ImageNet-100"
        typo_label = "ImageNet-100-Typo"

    # Create figure with reference styling
    fig, ax = plt.subplots(figsize=(5, 3))

    # Plot with reference colors and styling
    ax.plot(
        steps,
        [acc * 100 for acc in regular_accuracies],  # Convert to percentage
        color=COLOR_NORMAL,
        marker="o",
        markersize=4,
        label=normal_label,
        linestyle="-",
    )
    ax.plot(
        steps,
        [acc * 100 for acc in typo_accuracies],  # Convert to percentage
        color=COLOR_TYPO,
        marker="o",
        markersize=4,
        label=typo_label,
        linestyle="--",
    )

    # Customize plot appearance
    ax.set_xlabel("Number of Ablated Heads")
    ax.set_ylabel("Zeroshot Accuracy (%)")
    ax.set_title(f"{model_name_dict[model_short_name]}")

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

    save_dir.mkdir(parents=True, exist_ok=True)
    # Save plot
    save_path = save_dir / f"ablation_plot_{mode}.png"
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=150)
    save_path = save_dir / f"ablation_plot_{mode}.svg"
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    print(f"Plot saved to: {save_path}")

    return fig, ax


def plot_ablation_curve_grid(
    datasets: List[Dict], save_dir: Path, save_name: str = "ablation_grid"
) -> None:
    """
    Plot multiple ablation curves in a grid format with 3 curves per row.

    Args:
        datasets: List of dictionaries, each containing:
            - results_data: List[Dict] - The ablation results data
            - model_short_name: str - Short name for the model
            - mode: str - Mode identifier
        save_dir: Path to save the plots
        save_name: Base name for saved files
    """
    n_plots = len(datasets)
    n_cols = 3  # Three curves per row
    n_rows = (n_plots + n_cols - 1) // n_cols  # Calculate number of rows needed

    # Create figure with subplots in n_rows×3 grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    # Handle case where we have only one row or one plot
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, n_cols)
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for i, dataset in enumerate(datasets):
        results_data = dataset["results_data"]
        model_short_name = dataset["model_short_name"]
        mode = dataset["mode"]

        # Get the current axis
        ax = axes_flat[i]

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

        # Set labels based on model
        if model_short_name == "whylesionclip":
            normal_label = "ISIC2019"
            typo_label = "ISIC2019-Typo"
        else:
            normal_label = "ImageNet-100"
            typo_label = "ImageNet-100-Typo"

        # Plot with reference colors and styling
        ax.plot(
            steps,
            [acc * 100 for acc in regular_accuracies],  # Convert to percentage
            color=COLOR_NORMAL,
            marker="o",
            markersize=4,
            label=normal_label,
            linestyle="-",
        )
        ax.plot(
            steps,
            [acc * 100 for acc in typo_accuracies],  # Convert to percentage
            color=COLOR_TYPO,
            marker="o",
            markersize=4,
            label=typo_label,
            linestyle="--",
        )

        # Customize plot appearance
        ax.set_xlabel("Number of Ablated Heads")
        ax.set_ylabel("Zeroshot Accuracy (%)")
        ax.set_title(f"{model_name_dict[model_short_name]}")

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

    # Hide unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    # Save plot
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{save_name}.png"
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=150)
    save_path = save_dir / f"{save_name}.svg"
    plt.savefig(save_path, format="svg", bbox_inches="tight")
    save_path = save_dir / f"{save_name}.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Grid plot saved to: {save_path}")

    return fig, axes


if __name__ == "__main__":
    for model in ["vit-b", "vit-l", "vit-h", "vit-g", "vit-big-g", "whylesionclip"]:
        results_data = json.load(
            open(f"results/experiments/greedy_selection/{model}/results_cls.json")
        )
        plot_ablation_curve(
            results_data, model, "cls", Path(f"results/plots/greedy_selection/{model}")
        )

    datasets = [
        {
            "results_data": json.load(
                open(f"results/experiments/greedy_selection/{model}/results_cls.json")
            ),
            "model_short_name": model,
            "mode": "cls",
        }
        for model in [
            "vit-b",
            "vit-l",
            "vit-h",
            "vit-g",
            "vit-big-g",
            "whylesionclip",
        ]
    ]
    plot_ablation_curve_grid(
        datasets,
        Path("results/plots/greedy_selection/"),
    )
