# %%

### PLOT 1
from experiments.greedy_selection.greedy_selection import GreedySelectionExperiment

figs = []
axes = []

for model in ["vit-b", "vit-l", "vit-h", "vit-g", "whylesionclip"]:
    experiment = GreedySelectionExperiment(
        model_short_name=model,
        device="cpu",
        batch_size=16,
        num_workers=32,
        mode="cls",
    )

    results, heads = experiment.load_or_run_experiment()
    fig, ax = experiment.plot_results(results)
    figs.append(fig)
    axes.append(ax)
    del experiment, results, heads

# %%
from PIL import Image

im1 = Image.open("results/experiments/greedy_selection/vit-b/ablation_plot_cls.png")
im2 = Image.open("results/experiments/greedy_selection/vit-l/ablation_plot_cls.png")
im3 = Image.open("results/experiments/greedy_selection/vit-h/ablation_plot_cls.png")
im4 = Image.open("results/experiments/greedy_selection/vit-g/ablation_plot_cls.png")
im5 = Image.open(
    "results/experiments/greedy_selection/whylesionclip/ablation_plot_cls.png"
)

combined = Image.new("RGB", (im1.width * 2, im1.height * 3))
combined.paste(im1, (0, 0))
combined.paste(im2, (im1.width, 0))
combined.paste(im3, (0, im1.height))
combined.paste(im4, (im1.width, im1.height))
combined.paste(im5, (im1.width, im1.height * 2))
combined.save("results/experiments/greedy_selection/combined_ablation_plot_cls.png")

# %%
### PLOT 2
import json
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 3))

# Colors and markers for each model (darkest to lightest)
styles = {
    "vit-big-g": ("#29000F", "s"),  # Darkest burgundy
    "vit-g": ("#660025", "v"),  # Dark burgundy
    "vit-h": ("#A3003C", "D"),  # Medium burgundy
    "vit-l": ("#E00052", "^"),  # Light burgundy
    "vit-b": ("#FF1F71", "o"),  # Pink
}

for model in ["vit-b", "vit-l", "vit-h", "vit-g"]:
    accuracies = json.load(
        open(f"results/experiments/linear_probes/{model}/accuracies.json")
    )["typo_image_typo_label"]

    color, marker = styles[model]
    x_points = np.arange(0, 100, 100 / len(accuracies))
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

# %%
import matplotlib.pyplot as plt

plt.plot(accuracies)
plt.show()

# %%
