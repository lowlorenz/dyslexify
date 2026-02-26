# %%
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm

from dyslexify.cache.block import BlockResidual
from dyslexify.cache.cache import TypoLabeledResidualCache
from dyslexify.cache.collector import OpenClipActivationCollector
from dyslexify.config import DEVICE, MODELS
from dyslexify.dataset.ImageNet100 import ImageNet100
from dyslexify.probes import linear_probe_accuracy
from experiments.plot_config import COLOR_NORMAL, COLOR_TYPO, FIGSIZE


class LinearProbeExperiment:

    def __init__(
        self,
        model_short_name: str,
        device: str,
        batch_size: int = 128,
        num_workers: int = 16,
    ):
        self.model_short_name = model_short_name
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.device = device

        self.dataset, self.dataloader = self.setup_dataset(batch_size, num_workers)

        # Use a single collector since they're identical
        self.collector = OpenClipActivationCollector(
            self.model_name, self.pretrained, device
        )

    def setup_dataset(self, batch_size: int, num_workers: int):
        _, _, transform = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )

        dataset = ImageNet100(
            root="/datasets/imagenet-100-typo",
            split="val",
            preprocess=transform,
            position="random",
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return dataset, dataloader

    @torch.no_grad()
    def get_activation_cache(
        self,
    ) -> tuple[TypoLabeledResidualCache, TypoLabeledResidualCache]:
        normal_activations = None
        typo_activations = None

        for batch in tqdm(self.dataloader):
            img, typo_img, real_label, typo_label = batch

            img = img.to(self.device)
            typo_img = typo_img.to(self.device)

            # Get residual activation caches
            normal_cache = self.collector.get_typo_labeled_residual_cache(
                img, real_label, typo_label, cls_token_only=True
            )
            typo_cache = self.collector.get_typo_labeled_residual_cache(
                typo_img, real_label, typo_label, cls_token_only=True
            )

            if normal_activations is None:
                normal_activations = normal_cache
                typo_activations = typo_cache
            else:
                normal_activations.concatenate(normal_cache)
                typo_activations.concatenate(typo_cache)

        return normal_activations, typo_activations

    def probe_accuracy_per_mode(
        self,
        normal_activations: TypoLabeledResidualCache,
        typo_activations: TypoLabeledResidualCache,
    ) -> Dict[str, List[float]]:
        accuracies = {
            "normal_image_normal_label": self.probe_accuracy_over_residual(
                normal_activations.blocks,
                normal_activations.labels,
            ),
            "normal_image_typo_label": self.probe_accuracy_over_residual(
                normal_activations.blocks,
                typo_activations.typo_labels,
            ),
            "typo_image_normal_label": self.probe_accuracy_over_residual(
                typo_activations.blocks,
                normal_activations.labels,
            ),
            "typo_image_typo_label": self.probe_accuracy_over_residual(
                typo_activations.blocks,
                typo_activations.typo_labels,
            ),
        }
        return accuracies

    def probe_accuracy_over_residual(
        self,
        blocks: List[BlockResidual],
        labels: torch.Tensor,
    ) -> List[float]:
        probe_accuracy = []

        pbar = tqdm(total=len(blocks) * 2 + 1, desc="Probing accuracy over residual")

        results = linear_probe_accuracy(
            blocks[0].residual_pre,
            labels,
            num_classes=100,
            random_state=42,
            device=self.device,
            num_epochs=200,
        )
        probe_accuracy.append(results)
        pbar.update(1)

        for block in blocks:
            results = linear_probe_accuracy(
                block.residual_mid,
                labels,
                num_classes=100,
                random_state=42,
                device=self.device,
                num_epochs=200,
            )
            probe_accuracy.append(results)
            pbar.update(1)

            results = linear_probe_accuracy(
                block.residual_post,
                labels,
                num_classes=100,
                random_state=42,
                device=self.device,
                num_epochs=200,
            )
            probe_accuracy.append(results)
            pbar.update(1)

        return probe_accuracy

    def run_experiment(self):
        normal_activations, typo_activations = self.get_activation_cache()
        accuracies = self.probe_accuracy_per_mode(normal_activations, typo_activations)

        save_path = (
            Path("results/experiments/linear_probes")
            / self.model_short_name
            / "accuracies.json"
        )

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(accuracies, f, indent=2)

        return accuracies

    def load_accuracies(self) -> Dict[str, List[float]] | None:

        save_path = (
            Path("results/experiments/linear_probes")
            / self.model_short_name
            / "accuracies.json"
        )

        try:
            with open(save_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def load_or_run_experiment(self) -> Dict[str, List[float]]:
        if self.load_accuracies():
            return self.load_accuracies()
        else:
            return self.run_experiment()


if __name__ == "__main__":
    # for i, model_short_name in enumerate(MODELS.keys()):
    #     print(f"Running experiment for {model_short_name} ({i+1}/{len(MODELS.keys())})")
    #     experiment = LinearProbeExperiment(
    #         model_short_name=model_short_name,
    #         device=DEVICE,
    #         batch_size=128,
    #         num_workers=16,
    #     )
    #     accuracies = experiment.load_or_run_experiment()
    #     experiment.plot_accuracies(accuracies)

    experiment = LinearProbeExperiment(
        model_short_name="vit-l",
        device="cuda:1",
        batch_size=256,
        num_workers=16,
    )
    accuracies = experiment.load_or_run_experiment()


# Commented out code removed for clarity

# %%
