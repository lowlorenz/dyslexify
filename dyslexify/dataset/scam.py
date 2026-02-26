# %%
from dyslexify.dataset.base import BaseRealTypographicDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torch
from typing import Tuple
import os


class SCAM(BaseRealTypographicDataset):
    def __init__(self, root: str, preprocess=None):
        self.ds = load_dataset("BLISS-e-V/SCAM", split="train")
        self.scam = self.ds.filter(
            lambda batch: [t == "SCAM" for t in batch["type"]],
            batched=True,
            num_proc=os.cpu_count(),  # parallelize
        )

        self.no_scam = self.ds.filter(
            lambda batch: [t == "NoSCAM" for t in batch["type"]],
            batched=True,
            num_proc=os.cpu_count(),  # parallelize
        )

        self.templates = ["a photo of a {}."]
        self.classes = list(set(self.ds["object_label"]) | set(self.ds["attack_word"]))

        super().__init__(root, preprocess)

    def __len__(self) -> int:
        return len(self.scam)

    def _get_class_index(self, class_name: str) -> int:
        return self.classes.index(class_name)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        sample = self.scam[index]

        typo_img = sample["image"]
        no_typo_img = self.no_scam[index]["image"]

        if self.preprocess is not None:
            typo_img = self.preprocess(typo_img)
            no_typo_img = self.preprocess(no_typo_img)

        real_label = self._get_class_index(sample["object_label"])
        typo_label = self._get_class_index(sample["attack_word"])

        return no_typo_img, typo_img, real_label, typo_label


if __name__ == "__main__":
    dataset = SCAM(root="data/scam")
    print(dataset[0])

# %%
