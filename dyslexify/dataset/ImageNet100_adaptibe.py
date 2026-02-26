from pathlib import Path
import torch
import json
from PIL import Image
import os
from typing import Tuple, List, Dict, Any
from dyslexify.dataset.utils import _transform
from dyslexify.dataset.base import BaseTypographicDataset
from dyslexify.dataset.ImageNet100 import imagenet_100_templates, imagenet_100_classes


class ImageNet100Adaptive(BaseTypographicDataset):
    def __init__(
        self,
        root,
        split="train",
        preprocess=None,
        return_index=False,
        position="random",
        num_workers=None,
    ):
        # Load class mapping first
        with open(Path(root) / "data" / "Labels.json") as f:
            self.id_to_class = json.load(f)

        # Setup data directory
        self._data_dir = Path(root) / "data" / split

        # Load files and labels before calling super().__init__
        self._load_files_and_labels()

        # Setup templates
        self.templates = imagenet_100_templates

        # Now call the parent constructor
        super().__init__(
            root=root,
            split=split,
            preprocess=preprocess,
            return_index=return_index,
            position=position,
            num_workers=num_workers,
        )

    def _load_files_and_labels(self):
        """Load files and labels from the dataset directory."""
        # Load classes
        classes = []
        for dir in self._data_dir.iterdir():
            if not dir.is_dir():
                continue
            id = str(dir).split("/")[-1]
            class_i = self.id_to_class[id].split(",")[0]
            classes.append(class_i)

        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))

        # Load files and labels
        self._labels = []
        files = list(self._data_dir.rglob("*"))
        self._files = []

        for i in range(len(files)):
            if not files[i].is_file():
                continue
            self._files.append(files[i])

        for file in self._files:
            id = str(file).split("/")[-2]
            class_i = self.id_to_class[id].split(",")[0]
            self._labels.append(self.class_to_idx[class_i])

    def _load_dataset(self, split: str) -> Any:
        """Load the ImageNet100 dataset for the specified split."""
        # The dataset is loaded in _load_files_and_labels
        # This method is called by the base class but we handle loading in __init__
        return self._files

    def _get_valid_classes(self) -> List[str]:
        """Extract valid classes from the ImageNet100 dataset."""
        # All ImageNet100 classes are valid
        return self.classes

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        img_path = self._files[index]
        img = Image.open(img_path)

        return {"image": _transform(img), "class": self._labels[index]}

    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        img_path = self._files[index]
        img = Image.open(img_path)

        return {"image": _transform(img), "class": self.classes[self._labels[index]]}

    def _get_class_index(self, class_name: str) -> int:
        """Get the index of a class in the dataset."""
        return self.class_to_idx[class_name]

    def _get_dataset_size(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self._files)

    def _get_class_name_from_index(self, index: int) -> str:
        """Get the class name from a class index."""
        return self.classes[index]


if __name__ == "__main__":
    # Example usage with distributed processing
    dataset = ImageNet100Adaptive(
        root="/datasets/imagenet-100-typo",
        split="val",
        preprocess=None,  # No preprocessing to get raw images
        position="random",
        num_workers=4,  # Use 4 worker processes for generation
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"Sample classes: {dataset.classes[:5]}")

    # Test getting a sample
    if len(dataset) > 0:
        img, typo_img, real_label, typo_label = dataset[0]
        print(
            f"Sample - Real label: {real_label} ({dataset.get_class_name(real_label)}), Typo label: {typo_label} ({dataset.get_class_name(typo_label)})"
        )

    import code

    code.interact(local=dict(globals(), **locals()))
