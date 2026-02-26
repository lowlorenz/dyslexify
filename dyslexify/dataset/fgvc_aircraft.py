from pathlib import Path
import torch
import os
from PIL import Image
from typing import Tuple, List, Dict, Any
from torchvision.datasets import FGVCAircraft as TVFGVCAircraft
from dyslexify.dataset.utils import _transform
from dyslexify.dataset.base import BaseTypographicDataset


class FGVCAircraft(BaseTypographicDataset):
    def __init__(
        self,
        root,
        split="train",
        preprocess=None,
        return_index=False,
        position="random",
        num_workers=None,
        download=False,
    ):
        # Initialize the base TVFGVCAircraft dataset first
        self._tv_dataset = TVFGVCAircraft(
            root=root,
            split=split,
            transform=None,  # We'll handle transforms ourselves
            download=download,
        )

        # Setup data directory
        self._data_dir = Path(root)

        # Load files and labels before calling super().__init__
        self._load_files_and_labels()

        # Setup templates
        self.templates = ["a photo of a {}."]

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
        # Get classes from the TVFGVCAircraft dataset
        self.classes = self._tv_dataset.classes
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        # Get files and labels from the TVFGVCAircraft dataset
        self._files = self._tv_dataset._image_files
        self._labels = self._tv_dataset._labels

    def _load_dataset(self, split: str) -> Any:
        """Load the FGVCAircraft dataset for the specified split."""
        # The dataset is loaded in _load_files_and_labels
        # This method is called by the base class but we handle loading in __init__
        return self._files

    def _get_valid_classes(self) -> List[str]:
        """Extract valid classes from the FGVCAircraft dataset."""
        # All FGVCAircraft classes are valid
        return self.classes

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        img_path = self._files[index]
        img = Image.open(img_path).convert("RGB")

        return {"image": _transform(img), "class": self._labels[index]}

    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        img_path = self._files[index]
        img = Image.open(img_path).convert("RGB")

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
    dataset = FGVCAircraft(
        root="/datasets/fgvc-aircraft",
        split="test",
        preprocess=None,  # No preprocessing to get raw images
        position="random",
        download=True,  # Download the dataset if not present
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
