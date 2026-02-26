import torch
from pathlib import Path

# Load typographic scores for vit-b
data_path = Path(
    "results/experiments/typographic_scores/vit-b/typographic_scores_cls.pt"
)
typographic_scores = torch.load(data_path)

# Calculate mean and standard deviation
mean = typographic_scores.mean().item()
std = typographic_scores.std().item()

# Print results
print(f"vit_b typographic score statistics:")
print(f"Mean: {mean:.4f}")
print(f"Std:  {std:.4f}")


