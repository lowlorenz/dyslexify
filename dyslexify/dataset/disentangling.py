from dyslexify.dataset.base import BaseRealTypographicDataset
import json
from pathlib import Path
import torch
from PIL import Image
from typing import Tuple


class Disentangling(BaseRealTypographicDataset):
    def __init__(self, root: str, preprocess=None):
        super().__init__(root, preprocess)
        self.root = Path(root)
        annotations_path = self.root / "annotations.json"
        self.annotations = json.load(open(annotations_path))
        self.img_files = []
        self.labels = []
        self.typo_labels = []

        for file_name in self.annotations:
            self.img_files.append(self.root / file_name)

            self.labels.append(self.annotations[file_name]["true object"])
            self.typo_labels.append(
                self.annotations[file_name]["typographic attack label"]
            )

        self.classes = list(set(self.labels) | set(self.typo_labels))
        self.templates = ["a photo of a {}."]

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
    dataset = Disentangling(root="/datasets/disentangling")
    print(dataset[0])
    import code

    code.interact(local=dict(globals(), **locals()))
