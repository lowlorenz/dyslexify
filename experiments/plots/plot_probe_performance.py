from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from experiments.plot_config import COLOR_NORMAL, COLOR_TYPO
import json


def plot_accuracies(model_short_name: str, accuracies: Dict[str, List[float]]):
    """Plot typo probe performance with alternating background bands for attention and MLP layers."""
    plt.figure(figsize=(5, 3))

    # Get the typo probe accuracies
    typo_image_normal_label = accuracies["typo_image_normal_label"]
    typo_image_typo_label = accuracies["typo_image_typo_label"]

    full_points = len(typo_image_normal_label)
    x_points = np.arange(full_points)
    num_layers = (
        full_points - 1
    ) // 2  # Each layer has two points (pre and mid), excluding post probe

    # Add alternating background bands for each probe point up to the post probe
    for i in range(full_points - 1):  # Stop before the post probe point
        if i % 2 == 1:  # MLP layers (odd indices)
            plt.axvspan(
                x_points[i],
                x_points[i + 1],
                facecolor="none",  # No fill color
                color="gray",
                alpha=0.1,
                zorder=2,
            )
            # Add vertical lines at MLP layer boundaries
            plt.axvline(
                x=x_points[i],
                color="gray",
                linestyle="-",
                alpha=0.3,
                zorder=2,
                linewidth=1.2,
            )
            plt.axvline(
                x=x_points[i + 1],
                color="gray",
                linestyle="-",
                alpha=0.3,
                zorder=2,
                linewidth=1.2,
            )
        # No background for attention layers (even indices)

    plt.axvline(x=0, color="gray", linestyle="-", alpha=0.3, zorder=2, linewidth=1.0)

    # Plot both probe accuracy curves
    plt.plot(
        x_points,
        typo_image_normal_label,
        color=COLOR_NORMAL,  # Typo probe color
        label="Object Probe",
        marker="o",
        markersize=4,
        zorder=3,  # Ensure lines are drawn on top of hatching
    )
    plt.plot(
        x_points,
        typo_image_typo_label,
        color=COLOR_TYPO,  # Object probe color
        label="Typographic Probe",
        marker="o",
        markersize=4,
        zorder=3,  # Ensure lines are drawn on top of hatching
    )

    # Labels and legend
    if model_short_name == "WhyLesionClip":
        model_title_name = "WhyLesionClip"
    elif model_short_name == "WhyXrayClip":
        model_title_name = "WhyXrayClip"
    else:
        model_title_name = model_short_name.replace("vit", "ViT")
        model_title_name = model_title_name[:-1] + model_title_name[-1].upper()
    plt.xlabel(f"{model_title_name} Layer")
    plt.ylabel("Probe Accuracy")
    plt.legend(loc="upper left")

    # Set x-axis ticks to show layer numbers and move x-axis to top
    ax = plt.gca()
    # Set ticks at layer starts (even indices)
    layer_positions = list(
        range(0, full_points - 1, 2)
    )  # Every other position (pre-attention)
    layer_labels = [str(i) for i in range(num_layers)]  # Layer numbers
    ax.set_xticks(layer_positions)
    ax.set_xticklabels(layer_labels)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Move y-axis to the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Set x-axis limits to include exactly up to the last point
    ax.set_xlim(-0.5, full_points - 0.5)

    # Set y-axis limits
    ax.set_ylim(0, 1)

    # Remove plot edges
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Add horizontal grid lines only
    ax.yaxis.grid(True, linestyle="-", alpha=0.3)
    ax.xaxis.grid(False)

    plt.tight_layout()

    # Save plots in both formats
    save_dir_svg = Path("results") / "plots" / "linear_probes" / model_short_name
    save_dir_png = Path("results") / "plots" / "linear_probes" / model_short_name
    save_dir_svg.mkdir(parents=True, exist_ok=True)
    save_dir_png.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_dir_svg / "accuracies.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir_png / "accuracies.png",
        format="png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def plot_compare_typo_probe_performance(accuracies: Dict[str, List[float]]):

    plt.figure(figsize=(5, 3))

    # Colors and markers for each model (darkest to lightest)
    styles = {
        "vit-big-g": ("#29000F", "s"),  # Darkest burgundy
        "vit-g": ("#660025", "v"),  # Dark burgundy
        "vit-h": ("#A3003C", "D"),  # Medium burgundy
        "vit-l": ("#E00052", "^"),  # Light burgundy
        "vit-b": ("#FF1F71", "o"),  # Pink
    }

    for model, accuracies in accuracies.items():
        accuracies = accuracies[
            1::2
        ]  # only plot resid_mid points for visual simplicity
        color, marker = styles[model]
        x_points = np.arange(0, 100, 100 / len(accuracies) + 0.00001)
        plt.plot(
            x_points,
            accuracies,
            color=color,
            label=model.upper(),
            marker=marker,
            markersize=4,
        )

    plt.xlabel("Layer Position (%)")
    # Move y-axis to the right and update label
    ax = plt.gca()
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.ylabel("Probe Accuracy")
    plt.legend()
    plt.grid(True)

    # Remove plot edges
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set major ticks every 25%
    major_ticks = np.arange(0, 101, 25)
    plt.gca().set_xticks(major_ticks)

    # Set x-axis limits with some padding
    plt.xlim(-2, 102)

    # Add gridlines
    plt.grid(True, which="major", linestyle="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        Path("results")
        / "plots"
        / "linear_probes"
        / "compare_typo_probe_performance.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        Path("results")
        / "plots"
        / "linear_probes"
        / "compare_typo_probe_performance.png",
        format="png",
        bbox_inches="tight",
        dpi=150,
    )


def plot_compare_typo_real_probe_with_area(
    accuracies: Dict[str, Tuple[List[float], List[float]]],
):
    """Plot comparison between typo and real probe performance with shaded areas.

    Creates a plot showing the performance range for typo and real probes
    across all models, with shaded regions indicating the performance range.
    """
    plt.figure(figsize=(5, 3))

    # Colors for the regions and lines (using existing color scheme)
    typo_bg_color = "#FFB3D1"  # Light pink background for typo probes
    real_bg_color = "#D4C2DB"  # Light purple background for real/object probes
    typo_line_color = COLOR_TYPO  # Line color for typo probes (#FF5895)
    real_line_color = COLOR_NORMAL  # Line color for real/object probes (#A87EC1)

    # Prepare data for each model - use hook approach like plot_compare_typo_probe_performance
    typo_performances = []
    real_performances = []

    # Find the maximum number of points after applying hook filtering
    max_points = 0
    for model, (typo_accs, real_accs) in accuracies.items():
        filtered_length = len(typo_accs[1::2])  # only resid_mid points
        max_points = max(max_points, filtered_length)

    # Common x-axis points for interpolation (percentage-based like plot_compare_typo_probe_performance)
    x_common = np.arange(0, 100, 100 / max_points + 0.00001)

    for model, (typo_accs, real_accs) in accuracies.items():
        # Apply hook approach - only plot resid_mid points for visual simplicity
        typo_filtered = typo_accs[1::2]
        real_filtered = real_accs[1::2]

        # Create x points for this model
        x_points = np.arange(0, 100, 100 / len(typo_filtered) + 0.00001)

        # Interpolate to common length if needed
        if len(typo_filtered) != max_points:
            typo_interp = np.interp(x_common, x_points, typo_filtered)
            real_interp = np.interp(x_common, x_points, real_filtered)
        else:
            typo_interp = np.array(typo_filtered)
            real_interp = np.array(real_filtered)

        typo_performances.append(typo_interp)
        real_performances.append(real_interp)

    # Convert to numpy arrays for easier computation
    typo_performances = np.array(typo_performances)
    real_performances = np.array(real_performances)

    # Calculate min and max for each position
    typo_min = np.min(typo_performances, axis=0)
    typo_max = np.max(typo_performances, axis=0)
    real_min = np.min(real_performances, axis=0)
    real_max = np.max(real_performances, axis=0)

    # Plot the regions with shading
    plt.fill_between(
        x_common,
        typo_min,
        typo_max,
        color=typo_bg_color,
        alpha=0.5,
        label="Typographic Probes",
    )
    plt.fill_between(
        x_common,
        real_min,
        real_max,
        color=real_bg_color,
        alpha=0.5,
        label="Object Probes",
    )

    # Plot individual lines for each model
    for i in range(len(typo_performances)):
        plt.plot(
            x_common,
            typo_performances[i],
            color=typo_line_color,
            alpha=0.8,
            linewidth=1.5,
        )
        plt.plot(
            x_common,
            real_performances[i],
            color=real_line_color,
            alpha=0.8,
            linewidth=1.5,
        )

    plt.xlabel("Layer Position (%)")
    # Move y-axis to the right and update label (consistent with plot_compare_typo_probe_performance)
    ax = plt.gca()
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    plt.ylabel("Probe Accuracy")
    plt.legend()

    # Remove plot edges
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Set major ticks every 25%
    major_ticks = np.arange(0, 101, 25)
    ax.set_xticks(major_ticks)

    # Set x-axis limits with some padding
    plt.xlim(-2, 102)

    # Add gridlines
    plt.grid(True, which="major", linestyle="-", alpha=0.5)

    plt.tight_layout()

    # Save plots (consistent with existing save paths)
    plt.savefig(
        Path("results")
        / "plots"
        / "linear_probes"
        / "compare_typo_real_probe_with_area.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        Path("results")
        / "plots"
        / "linear_probes"
        / "compare_typo_real_probe_with_area.png",
        format="png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


if __name__ == "__main__":
    for model_short_name in [
        "vit-b",
        "vit-l",
        "vit-h",
        "vit-g",
        "vit-big-g",
        "WhyLesionClip",
        "WhyXrayClip",
    ]:
        accuracies = json.load(
            open(
                f"results/experiments/linear_probes/{model_short_name}/accuracies.json"
            )
        )
        plot_accuracies(model_short_name, accuracies)

    accuracies = {
        model: json.load(
            open(f"results/experiments/linear_probes/{model}/accuracies.json")
        )["typo_image_typo_label"]
        for model in ["vit-b", "vit-l", "vit-h", "vit-g", "vit-big-g"]
    }
    plot_compare_typo_probe_performance(accuracies)

    accuracies = {
        model: (
            json.load(
                open(f"results/experiments/linear_probes/{model}/accuracies.json")
            )["typo_image_typo_label"],
            json.load(
                open(f"results/experiments/linear_probes/{model}/accuracies.json")
            )["normal_image_normal_label"],
        )
        for model in ["vit-b", "vit-l", "vit-h", "vit-g", "vit-big-g"]
    }
    plot_compare_typo_real_probe_with_area(accuracies)
