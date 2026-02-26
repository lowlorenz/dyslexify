from PIL import Image
import torch
from dyslexify.dataset.utils import _transform
from typing import Tuple, List, Dict, Any
from datasets import load_dataset
from dyslexify.dataset.base import BaseTypographicDataset


class UnsplashTypographicDataset(BaseTypographicDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        preprocess=None,
        return_index: bool = False,
        position: str = "random",
        num_workers: int = None,
    ):
        super().__init__(
            root=root,
            split=split,
            preprocess=preprocess,
            return_index=return_index,
            position=position,
            num_workers=num_workers,
        )

    def _load_dataset(self, split: str) -> Any:
        """Load the Unsplash dataset for the specified split."""
        return load_dataset("wtcherr/unsplash_10k_canny", split=split)

    def _get_valid_classes(self) -> List[str]:
        """Extract valid classes from the Unsplash dataset."""
        return [text for text in self.dataset["text"] if 6 <= len(text) <= 18]

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        item = self.dataset[index]
        return {
            "image": item["image"],
            "class": 0,
        }  # Unsplash originally doesn't have labels

    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        item = self.dataset[index]
        return {"image": item["image"], "class": item["text"]}

    def _get_class_index(self, class_name: str) -> int:
        """Get the index of a class in the dataset."""
        return self.dataset["text"].index(class_name)

    def _get_dataset_size(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self.dataset)

    def _get_class_name_from_index(self, index: int) -> str:
        """Get the class name from a class index."""
        return self.dataset["text"][index]


if __name__ == "__main__":
    # Example usage with distributed processing
    dataset = UnsplashTypographicDataset(
        root="/datasets/unsplash_10k_canny",
        split="train",
        preprocess=None,
        return_index=False,
        position="random",
    )
    import code

    code.interact(local=dict(globals(), **locals()))
