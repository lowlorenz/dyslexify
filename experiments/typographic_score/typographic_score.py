import open_clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dyslexify.cache.collector import OpenClipActivationCollector
from dyslexify.config import MODELS
from dyslexify.dataset.melanoma import Melanoma
from dyslexify.dataset.unsplash import UnsplashTypographicDataset
from dyslexify.cache.block import BlockAttention
import math
import einops
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from experiments.plot_config import FIGSIZE, COLOR_TYPO, COLOR_NORMAL
import seaborn as sns


class TypographicScoreExperiment:

    def __init__(
        self,
        model_short_name: str,
        device: str,
        batch_size: int = 128,
        num_workers: int = 16,
        mode: str = "cls",
    ):
        self.model_short_name = model_short_name
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        if mode not in ["cls", "spatial"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode

        # load model
        _, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )
        self.collector = OpenClipActivationCollector(
            self.model_name, self.pretrained, device
        )

        self.dataset, self.dataloader = self.setup_dataset()

    def setup_dataset(self):
        # load dataset
        # if self.model_short_name == "whylesionclip":
        #     dataset = MelanomaTypographicDataset(
        #         root="/datasets/melanoma_cancer_dataset_typo",
        #         split="train",
        #         preprocess=self.preprocess,
        #         position="bottom",
        #     )
        # else:
        dataset = UnsplashTypographicDataset(
            root="/datasets/unsplash_10k_canny",
            split="train",
            preprocess=self.preprocess,
            position="bottom",
        )

        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return dataset, dataloader

    def load_or_run_experiment(self):
        save_path = Path(
            f"results/experiments/typographic_scores/{self.model_short_name}"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"typographic_scores_{self.mode}.pt"
        if save_path.exists():
            typographic_scores = torch.load(save_path)
            return typographic_scores
        else:
            return self.run_experiment()

    def run_experiment(self):
        attention_cache = self.collect_mean_attention_pattern()
        typographic_scores = self.calculate_typographic_score(attention_cache)

        save_path = Path(
            f"results/experiments/typographic_scores/{self.model_short_name}"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"typographic_scores_{self.mode}.pt"
        torch.save(typographic_scores, save_path)
        print(f"Saved typographic scores to {save_path}")

        return typographic_scores

    @torch.no_grad()
    @torch.autocast("cuda")
    def collect_mean_attention_pattern(self):
        attention_cache = None

        num_elems = 0
        for i, batch in enumerate(tqdm(self.dataloader)):
            img, typo_img, real_label, typo_label = batch
            typo_img = typo_img.to(self.device)

            batch_size = typo_img.shape[0]
            num_elems += batch_size

            # Get attention cache only
            if self.mode == "cls":
                current_cache = self.collector.get_attention_cache(
                    typo_img, cls_token_only=True
                )
            else:
                # mean over spatial tokens
                current_cache = self.collector.get_attention_cache(
                    typo_img, cls_token_only=False
                )
                for block in current_cache.blocks:
                    block.attn_pattern = block.attn_pattern[:, :, 1:, :].mean(dim=2)

            for block in current_cache.blocks:
                block.attn_pattern = block.attn_pattern.sum(dim=0)

            if attention_cache is None:
                attention_cache = current_cache
                continue

            # Add attention patterns
            for current_block, add_block in zip(
                attention_cache.blocks, current_cache.blocks
            ):
                current_block.attn_pattern += add_block.attn_pattern

        # Normalize by number of elements
        for block in attention_cache.blocks:
            block.attn_pattern = block.attn_pattern / num_elems

        return attention_cache

    def calculate_typographic_score(self, attention_cache):
        typographic_scores = []
        for block in attention_cache.blocks:
            # block.attn_pattern
            # - shape: (num_heads, num_tokens), only contains the cls token attention pattern

            # drop cls token
            pattern = block.attn_pattern[:, 1:]

            # reshape to 2D grid
            pattern = einops.rearrange(
                pattern,
                "heads (w1 w2) -> heads w1 w2",
                w1=int(math.sqrt(pattern.shape[1])),
            )

            # calculate typographic score
            typographic_attn = einops.reduce(  # get last 2 rows attention
                pattern[:, -2:, :],
                "heads w1 w2 -> heads",
                reduction="sum",
            )

            general_attn = einops.reduce(  # get general attention
                pattern,
                "heads w1 w2 -> heads",
                reduction="sum",
            )
            typographic_score = typographic_attn / general_attn

            typographic_scores.append(typographic_score)

        return torch.stack(typographic_scores).cpu()


if __name__ == "__main__":
    experiment = TypographicScoreExperiment(
        model_short_name="vit-b",
        device="cuda:0",
        batch_size=256,
        num_workers=32,
        mode="spatial",
    )
    # experiment = TypographicScoreExperiment(
    #     model_short_name="whylesionclip",
    #     device="cuda:3",
    #     batch_size=16,
    #     num_workers=64,
    #     mode="cls",
    # )
    typographic_scores = experiment.load_or_run_experiment()
    # experiment = TypographicScoreExperiment(
    #     model_short_name="vit-b",
    #     device="cuda:7",
    #     batch_size=16,
    #     num_workers=64,
    #     mode="cls",
    # )
    # typographic_scores = experiment.load_or_run_experiment()
    # experiment.plot_typographic_scores(typographic_scores)
    # for model_short_name in MODELS.keys():
    #     experiment = TypographicScoreExperiment(
    #         model_short_name=model_short_name,
    #         device="cuda:6",
    #         batch_size=16,
    #         num_workers=64,
    #     )
    #     typographic_scores = experiment.load_or_run_experiment()
    #     experiment.plot_typographic_scores(typographic_scores)
