import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from experiments.plot_config import COLOR_TYPO


def format_model_title(model: str) -> str:
    """Format model name for display in titles."""
    if model == "whylesionclip" or model == "WhyLesionClip":
        return "WhyLesionClip"
    elif model == "whyxrayclip" or model == "WhyXrayClip":
        return "WhyXrayClip"
    else:
        # Handle ViT models
        title = model.replace("vit", "ViT")
        if (
            title.endswith("-b")
            or title.endswith("-l")
            or title.endswith("-h")
            or title.endswith("-g")
        ):
            title = title[:-1] + title[-1].upper()
        return title


def plot_typographic_scores(model, typographic_scores, mode="cls"):
    """Create a heatmap showing typographic scores for all layers and heads."""
    # Convert typographic scores to numpy array
    scores_array = typographic_scores.numpy()  # Shape: (num_heads, num_layers)
    num_layers, num_heads = scores_array.shape

    # Get min and max values for colorbar normalization
    vmin = 0  # Start from 0 for consistent scaling
    vmax = scores_array.max()

    # Calculate figure size based on number of heads and layers
    # Use different base sizes for width and height to make boxes wider
    base_width = 0.5  # inches per cell width
    base_height = 0.5  # inches per cell height
    width = base_width * num_heads  # width now depends on number of heads
    height = base_height * num_layers  # height now depends on number of layers
    plt.figure(figsize=(width, height))

    # Create heatmap with values displayed
    ax = sns.heatmap(
        scores_array * 100,  # No need to transpose since we want layers on y-axis
        cmap="GnBu",
        annot=True,  # Show values
        fmt=".1f",  # Format to 1 decimal place
        cbar=False,  # Don't show colorbar
        vmin=vmin * 100,  # Start from 0
        vmax=vmax * 100,  # Use data max
        annot_kws={"size": 10},  # Normal font size for annotations
        xticklabels=[f"L{i}" for i in range(num_heads)],  # Head labels on x-axis
        yticklabels=[f"H{i}" for i in range(num_layers)],  # Layer labels on y-axis
    )

    # Move y-axis to right
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")

    # Customize the plot labels
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Head", fontsize=12)

    # Adjust labels with normal font size
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10, rotation=0)

    # Save plots in both formats
    save_dir_svg = Path("results") / "plots" / "typographic_scores" / model
    save_dir_png = Path("results") / "plots" / "typographic_scores" / model
    save_dir_svg.mkdir(parents=True, exist_ok=True)
    save_dir_png.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_dir_svg / f"typographic_scores_{mode}.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir_png / f"typographic_scores_{mode}.png",
        format="png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def plot_typographic_scores_threshold(
    model, linear_probe_accuracy, typographic_scores, mode="cls"
):
    """Create a heatmap showing typographic scores for all layers and heads."""
    # Convert typographic scores to numpy array
    scores_array = typographic_scores.numpy()  # Shape: (num_heads, num_layers)
    num_layers, num_heads = scores_array.shape

    # Get min and max values for colorbar normalization
    vmin = 0  # Start from 0 for consistent scaling
    vmax = scores_array.max()

    mean = scores_array.mean()
    std = scores_array.std()

    threshold = mean + 2 * std

    scores_array[scores_array < threshold] = 0
    scores_array[scores_array >= threshold] = 1

    # Calculate figure size based on number of heads and layers
    # Use different base sizes for width and height to make boxes wider
    base_width = 0.5  # inches per cell width
    base_height = 0.5  # inches per cell height
    width = base_width * num_heads  # width now depends on number of heads
    height = base_height * num_layers  # height now depends on number of layers
    plt.figure(figsize=(width, height))

    # Create heatmap with values displayed
    ax = sns.heatmap(
        scores_array * 100,  # No need to transpose since we want layers on y-axis
        cmap="Greys",
        cbar=False,  # Don't show colorbar
        xticklabels=[f"L{i}" for i in range(num_heads)],  # Head labels on x-axis
        yticklabels=[f"H{i}" for i in range(num_layers)],  # Layer labels on y-axis
    )

    # Create a secondary y-axis for linear probe accuracy
    ax2 = ax.twinx()

    # overlay linear probe accuracy on secondary axis
    x_coords = [
        i * 0.5 for i in range(len(linear_probe_accuracy))
    ]  # Center at tick positions
    ax2.plot(
        x_coords,
        linear_probe_accuracy,
        color=COLOR_TYPO,
        marker="o",
        markersize=4,
        linewidth=2,
    )

    # Configure secondary y-axis
    ax2.set_ylabel("Probe Accuracy", fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_label_position("left")

    # Keep y-axis (Head labels) on the right
    ax.yaxis.set_ticks_position("right")
    ax.yaxis.set_label_position("right")

    # Add legend for the probe accuracy line
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=COLOR_TYPO,
            marker="o",
            markersize=4,
            linewidth=2,
            label="Typographic Probe",
        ),
        Rectangle(
            (0, 0),
            0,
            0,
            facecolor="black",
            edgecolor="black",
            label="Heads with $T_{i,\ell} > 2\sigma _T + \mu _T$",
        ),
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=10, frameon=True)

    # Customize the plot labels
    plt.xlabel("Layer", fontsize=12)
    ax.set_ylabel("Head", fontsize=12)
    ax.yaxis.label.set_color("black")

    # Adjust labels with normal font size
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10, rotation=0)

    # Save plots in both formats
    save_dir_svg = Path("results") / "plots" / "typographic_scores" / model
    save_dir_png = Path("results") / "plots" / "typographic_scores" / model
    save_dir_svg.mkdir(parents=True, exist_ok=True)
    save_dir_png.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_dir_svg / f"typographic_scores_{mode}_threshold.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.savefig(
        save_dir_png / f"typographic_scores_{mode}_threshold.png",
        format="png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def plot_typographic_scores_combined(
    model, linear_probe_accuracy, typographic_scores, mode="cls"
):
    """Create a combined plot with typographic scores on top and thresholded scores on bottom."""
    # Convert typographic scores to numpy array
    original_scores = typographic_scores.numpy()  # Shape: (num_heads, num_layers)
    num_layers, num_heads = original_scores.shape

    # Get min and max values for colorbar normalization
    vmin = 0  # Start from 0 for consistent scaling
    vmax = original_scores.max()

    # Calculate threshold for bottom plot
    mean = original_scores.mean()
    std = original_scores.std()
    threshold = mean + 2 * std

    # Create thresholded version
    thresholded_array = original_scores.copy()
    thresholded_array[thresholded_array < threshold] = 0
    thresholded_array[thresholded_array >= threshold] = 1

    # Calculate figure size based on number of heads and layers
    base_width = 0.5  # inches per cell width
    base_height = 0.5  # inches per cell height
    width = base_width * num_heads
    height = base_height * num_layers * 2  # Double height for two plots

    # Create figure with subplots
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(width, height))

    # Top plot: Original typographic scores
    sns.heatmap(
        original_scores * 100,
        cmap="GnBu",
        annot=True,
        fmt=".1f",
        cbar=False,
        vmin=vmin * 100,
        vmax=vmax * 100,
        annot_kws={"size": 10},
        xticklabels=[f"L{i}" for i in range(num_heads)],
        yticklabels=[f"H{i}" for i in range(num_layers)],
        ax=ax_top,
    )

    # Top plot styling
    ax_top.yaxis.set_ticks_position("right")
    ax_top.yaxis.set_label_position("right")
    ax_top.set_xlabel("")  # Remove xlabel from top plot
    ax_top.set_ylabel("Head", fontsize=12)
    model_title = format_model_title(model)
    ax_top.set_title(f"{model_title} - Typographic Scores", fontsize=14, pad=10)
    plt.setp(
        ax_top.get_xticklabels(), fontsize=10, visible=False
    )  # Hide x labels on top plot
    plt.setp(ax_top.get_yticklabels(), fontsize=10, rotation=0)

    # Bottom plot: Thresholded scores with probe accuracy overlay
    sns.heatmap(
        thresholded_array * 100,
        cmap="Greys",
        cbar=False,
        xticklabels=[f"L{i}" for i in range(num_heads)],
        yticklabels=[f"H{i}" for i in range(num_layers)],
        ax=ax_bottom,
    )

    # Create a secondary y-axis for linear probe accuracy on bottom plot
    ax_probe = ax_bottom.twinx()

    # Overlay linear probe accuracy on secondary axis
    x_coords = [i * 0.5 for i in range(len(linear_probe_accuracy))]
    ax_probe.plot(
        x_coords,
        linear_probe_accuracy,
        color=COLOR_TYPO,
        marker="o",
        markersize=4,
        linewidth=2,
    )

    # Configure secondary y-axis
    ax_probe.set_ylabel("Probe Accuracy", fontsize=12)
    ax_probe.set_ylim(0, 1)
    ax_probe.yaxis.set_ticks_position("left")
    ax_probe.yaxis.set_label_position("left")

    # Bottom plot styling
    ax_bottom.yaxis.set_ticks_position("right")
    ax_bottom.yaxis.set_label_position("right")
    ax_bottom.set_xlabel("Layer", fontsize=12)
    ax_bottom.set_ylabel("Head", fontsize=12)
    ax_bottom.set_title("Thresholded Scores", fontsize=14, pad=10)
    plt.setp(ax_bottom.get_xticklabels(), fontsize=10)
    plt.setp(ax_bottom.get_yticklabels(), fontsize=10, rotation=0)

    # Add legend for the probe accuracy line
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    legend_elements = [
        Line2D(
            [0],
            [0],
            color=COLOR_TYPO,
            marker="o",
            markersize=4,
            linewidth=2,
            label="Typographic Probe",
        ),
        Rectangle(
            (0, 0),
            0,
            0,
            facecolor="black",
            edgecolor="black",
            label="Heads with $T_{i,\ell} > 2\sigma _T + \mu _T$",
        ),
    ]
    ax_probe.legend(
        handles=legend_elements, loc="upper left", fontsize=10, frameon=True
    )

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save plots in both formats
    save_dir_svg = Path("results") / "plots" / "typographic_scores" / model
    save_dir_png = Path("results") / "plots" / "typographic_scores" / model
    save_dir_svg.mkdir(parents=True, exist_ok=True)
    save_dir_png.mkdir(parents=True, exist_ok=True)

    model_title = format_model_title(model)
    plt.savefig(
        save_dir_svg / f"{model_title}_typographic_scores_{mode}_combined.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    for model in [
        "vit-b",
        "vit-l",
        "vit-h",
        "vit-g",
        "vit-big-g",
        "whylesionclip",
        "whyxrayclip",
    ]:
        # Map model names for different directory structures
        typo_scores_model = model  # typographic scores use lowercase
        probe_model = (
            "WhyLesionClip"
            if model == "whylesionclip"
            else "WhyXrayClip" if model == "whyxrayclip" else model
        )

        path = Path(
            f"results/experiments/typographic_scores/{typo_scores_model}/typographic_scores_cls.pt"
        )
        linear_probe_accuracy = json.load(
            open(f"results/experiments/linear_probes/{probe_model}/accuracies.json")
        )["typo_image_typo_label"]
        # linear_probe_accuracy = linear_probe_accuracy[1::2]
        typographic_scores = torch.load(path).T
        plot_typographic_scores(model, typographic_scores, mode="cls")
        plot_typographic_scores_combined(
            model, linear_probe_accuracy, typographic_scores, mode="cls"
        )
        plot_typographic_scores_threshold(
            model, linear_probe_accuracy, typographic_scores, mode="cls"
        )
