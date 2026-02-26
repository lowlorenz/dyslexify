# %%
from dyslexify.dataset.base import BaseRealTypographicDataset
from PIL import Image
import torch
from typing import Tuple
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class RTA100(BaseRealTypographicDataset):
    def __init__(self, root: str, preprocess=None):

        self.img_files = []
        self.labels = []
        self.typo_labels = []

        for file in os.listdir(root):
            self.img_files.append(os.path.join(root, file))
            typo_label = (
                file.split("_")[1]
                .replace("text=", "")
                .replace(".jpg", "")
                .replace(".png", "")
            )
            label = file.split("_")[0].replace("label=", "")

            self.labels.append(label)
            self.typo_labels.append(typo_label)

        self.classes = list(set(self.labels) | set(self.typo_labels))

        self.templates = ["a photo of a {}."]
        super().__init__(root, preprocess)

    def __len__(self) -> int:
        return len(self.img_files)

    def _get_class_index(self, class_name: str) -> int:
        return self.classes.index(class_name)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        img = Image.open(self.img_files[index])

        if self.preprocess is not None:
            img = self.preprocess(img)

        real_label = self._get_class_index(self.labels[index])
        typo_label = self._get_class_index(self.typo_labels[index])

        return img, img, real_label, typo_label


if __name__ == "__main__":
    dataset = RTA100(
        root="/datasets/rta100",
        preprocess=transforms.Compose([transforms.ToTensor()]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=16,
    )
    print(next(iter(dataloader)))

# %%
