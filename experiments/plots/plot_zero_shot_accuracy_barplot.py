import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Add workspace root to path for imports
import sys

workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from experiments.plot_config import COLOR_TYPO, COLOR_NORMAL


def get_datasets_for_model(model: str):
    """Get appropriate datasets for each model."""
    if model == "whylesionclip":
        return ["ISIC2019Binary", "Melanoma", "BCN20k", "HAM10k"]
    else:
        return [
            "ImageNet100",
            "Food101",
            "FGVCAircraft",
            "RTA100",
            "Disentangling",
            "Paint",
        ]


COLOR_TYPO_LIGHT = "lightgray"
COLOR_NORMAL_LIGHT = "white"


def load_results(model: str):
    """Load evaluation results for a model."""
    path = (
        workspace_root / "results" / "experiments" / "eval" / model / "results_cls.json"
    )
    with open(path, "r") as f:
        return json.load(f)


def create_barplot():
    """Create zero-shot accuracy bar plot."""
    models = ["whylesionclip"]
    model_labels = ["WhyLesionCLIP"]

    fig, axes = plt.subplots(1, len(models), figsize=(8 / 1.3, 5 / 1.3))

    for idx, (model, label) in enumerate(zip(models, model_labels)):
        ax = axes

        # Load data
        results = load_results(model)
        datasets = get_datasets_for_model(model)
        available_datasets = [d for d in datasets if d in results]

        # Extract values and convert to percentages
        base_acc = [results[d]["accuracy"] * 100 for d in available_datasets]
        typo_acc = [results[d]["typo_accuracy"] * 100 for d in available_datasets]
        dislex_acc = [
            results[d]["dislexified_accuracy"] * 100 for d in available_datasets
        ]
        dislex_typo_acc = [
            results[d]["dislexified_typo_accuracy"] * 100 for d in available_datasets
        ]

        # Calculate deltas
        delta_normal = [d - b for d, b in zip(dislex_acc, base_acc)]
        delta_typo = [d - b for d, b in zip(dislex_typo_acc, typo_acc)]

        # Plot setup
        x = np.arange(len(available_datasets))
        width = 0.32
        margin = 0.05
        # Plot delta overlays
        for i, (dn, dt, ba, ta) in enumerate(
            zip(delta_normal, delta_typo, base_acc, typo_acc)
        ):
            # Non-typo delta - use exact same positioning as base bar
            if dn != 0:
                color = "#2E8B57" if dn > 0 else "#DC143C"
                hatch = "///" if dn > 0 else "\\\\\\"
                bottom = ba if dn > 0 else ba + dn
                height = abs(dn)

                ax.bar(
                    x[i] - width / 2 - margin / 2,
                    height,
                    width,
                    bottom=bottom,
                    # facecolor="none",
                    facecolor=color,
                    # hatch=hatch,
                    edgecolor=color,
                    linewidth=2,
                )

                # Annotate
                y_pos = ba + dn + (1.0 if dn > 0 else -1.0)
                ax.text(
                    x[i] - width / 2 - margin / 2,
                    y_pos,
                    f"{dn:+.1f}%",
                    ha="center",
                    va="bottom" if dn > 0 else "top",
                    fontsize=10,
                    color=color,
                    weight="bold",
                )

            # Typo delta - use exact same positioning as base bar
            if dt != 0:
                color = "#2E8B57" if dt > 0 else "#DC143C"
                hatch = "///" if dt > 0 else "\\\\\\"
                bottom = ta if dt > 0 else ta + dt
                height = abs(dt)

                ax.bar(
                    x[i] + width / 2 + margin / 2,
                    height,
                    width,
                    bottom=bottom,
                    # hatch=hatch,
                    # facecolor="none",
                    facecolor=color,
                    edgecolor=color,
                    linewidth=2,
                )

                # Annotate
                y_pos = ta + dt + (1.0 if dt > 0 else -1.0)
                ax.text(
                    x[i] + width / 2 + margin / 2,
                    y_pos,
                    f"{dt:+.1f}%",
                    ha="center",
                    va="bottom" if dt > 0 else "top",
                    fontsize=10,
                    color=color,
                    weight="bold",
                )
        # Plot base bars (outline only)
        ax.bar(
            x - width / 2 - margin / 2,
            base_acc,
            width,
            # facecolor="none",
            facecolor=COLOR_NORMAL_LIGHT,
            edgecolor="black",
            linewidth=1,
        )
        ax.bar(
            x + width / 2 + margin / 2,
            typo_acc,
            width,
            # facecolor="none",
            facecolor=COLOR_TYPO_LIGHT,
            edgecolor="black",
            linewidth=1,
        )

        # Rename datasets for display
        display_names = [
            d.replace("ISIC2019Binary", "ISIC2019") for d in available_datasets
        ]

        # Styling
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=0, ha="center", fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.3)

        # Add legend
        legend_elements = [
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="none",
                edgecolor=COLOR_NORMAL,
                linewidth=2,
                label="Standard Dataset",
            ),
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="none",
                edgecolor=COLOR_TYPO,
                linewidth=2,
                label="Typo Dataset",
            ),
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="none",
                # hatch="///",
                edgecolor="#2E8B57",
                linewidth=1,
                label="Dislexify Change",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    # Save
    save_dir = workspace_root / "results" / "plots" / "zero_shot_accuracy"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_dir / "zero_shot_accuracy_barplot.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        save_dir / "zero_shot_accuracy_barplot.svg",
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Plots saved to {save_dir}")
    plt.show()


def create_dumbbell_plot():
    """Create dumbbell plot showing accuracy changes."""
    models = ["whylesionclip"]
    model_labels = ["WhyLesionCLIP"]

    fig, ax = plt.subplots(1, 1, figsize=(8 / 1.3, 5 / 1.3))

    for model, label in zip(models, model_labels):
        # Load data
        results = load_results(model)
        datasets = get_datasets_for_model(model)
        available_datasets = [d for d in datasets if d in results][::-1]

        # Extract values and convert to percentages
        base_acc = [results[d]["accuracy"] * 100 for d in available_datasets]
        typo_acc = [results[d]["typo_accuracy"] * 100 for d in available_datasets]
        dislex_acc = [
            results[d]["dislexified_accuracy"] * 100 for d in available_datasets
        ]
        dislex_typo_acc = [
            results[d]["dislexified_typo_accuracy"] * 100 for d in available_datasets
        ]

        # Create grouped dataset names (non-typo and typo versions)
        grouped_datasets = []
        grouped_base_acc = []
        grouped_dislex_acc = []

        # Add non-typographic datasets
        for i, dataset in enumerate(available_datasets):
            display_name = dataset.replace("ISIC2019Binary", "ISIC2019")
            grouped_datasets.append(display_name)
            grouped_base_acc.append(base_acc[i])
            grouped_dislex_acc.append(dislex_acc[i])

        # Add typographic datasets with -typo suffix
        for i, dataset in enumerate(available_datasets):
            display_name = dataset.replace("ISIC2019Binary", "ISIC2019") + "-typo"
            grouped_datasets.append(display_name)
            grouped_base_acc.append(typo_acc[i])
            grouped_dislex_acc.append(dislex_typo_acc[i])

        # Setup y positions for grouped datasets with gap between groups
        n_datasets = len(available_datasets)
        y_positions = []

        # Non-typographic datasets (top group)
        for i in range(n_datasets):
            y_positions.append(i)

        # Add gap and typographic datasets (bottom group)
        gap = 0.5
        for i in range(n_datasets):
            y_positions.append(i + n_datasets + gap)

        y_pos = np.array(y_positions)

        # Add visual separator line between groups
        separator_y = n_datasets - 0.5 + gap / 2
        ax.axhline(
            y=separator_y, color="lightgray", linestyle="--", alpha=0.7, linewidth=1
        )

        # Add group labels
        ax.text(
            -5,
            n_datasets / 2 - 0.5,
            "Standard",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            weight="bold",
        )
        ax.text(
            -5,
            n_datasets + gap + n_datasets / 2 - 0.5,
            "Typographic",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
            weight="bold",
        )

        # Plot dumbbell connections (lines)
        for i, (ba, da) in enumerate(zip(grouped_base_acc, grouped_dislex_acc)):
            ax.plot(
                [ba, da], [y_pos[i], y_pos[i]], color="#2E8B57", linewidth=5, alpha=0.7
            )

        # Plot base accuracy points
        ax.scatter(
            grouped_base_acc,
            y_pos,
            color="white",
            edgecolor="black",
            s=80,
            zorder=3,
            label="Base Model",
        )

        # Plot dislexified accuracy points
        ax.scatter(
            grouped_dislex_acc,
            y_pos,
            color="black",
            edgecolor="black",
            s=80,
            zorder=3,
            label="Dislexified Model",
        )

        # Add delta annotations
        for i, (ba, da) in enumerate(zip(grouped_base_acc, grouped_dislex_acc)):
            delta = da - ba
            if delta != 0:
                color = "#2E8B57" if delta > 0 else "#DC143C"
                ax.text(
                    max(ba, da) + 2,
                    y_pos[i],
                    f"{delta:+.1f}%",
                    va="center",
                    fontsize=9,
                    color=color,
                    weight="bold",
                )

        # Use grouped dataset names for display
        display_names = grouped_datasets

        # Styling
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Accuracy (%)", fontsize=12)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=10)
        ax.yaxis.tick_right()
        ax.set_xlim(-10, 100)
        ax.grid(True, axis="x", alpha=0.3)

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=8,
                label="Base Model",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=8,
                label="Dislexified Model",
            ),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save
    save_dir = workspace_root / "results" / "plots" / "zero_shot_accuracy"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_dir / "zero_shot_accuracy_dumbbell.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(
        save_dir / "zero_shot_accuracy_dumbbell.svg",
        format="svg",
        bbox_inches="tight",
        facecolor="white",
    )

    print(f"Dumbbell plot saved to {save_dir}")
    plt.show()


if __name__ == "__main__":
    create_barplot()
    create_dumbbell_plot()
