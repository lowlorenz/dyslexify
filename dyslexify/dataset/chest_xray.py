import os
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from dyslexify.dataset.base import BaseTypographicDataset


class ChestXRay(BaseTypographicDataset):
    """
    Chest X-Ray dataset for pneumonia detection with typographic attack support.

    Expected directory structure:
        root/
          train/
            NORMAL/
              *.jpeg
            PNEUMONIA/
              *.jpeg
          test/
            NORMAL/
              *.jpeg
            PNEUMONIA/
              *.jpeg
          val/
            NORMAL/
              *.jpeg
            PNEUMONIA/
              *.jpeg
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def __init__(self, *args, **kwargs):
        self.classes = ["Normal", "Pneumonia"]
        self.templates = ["A photo of a {} chest x-ray."]
        super().__init__(*args, **kwargs)

    def _load_dataset(self, split: str) -> Any:
        """
        Scan the filesystem and index all samples for the provided split.
        Returns a list of tuples (image_path: Path, class_name: str).
        """
        split_dir = Path(self.root) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Discover classes from subdirectories under the split directory
        class_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class subdirectories found under {split_dir}")

        self._class_names = [d.name.capitalize() for d in class_dirs]
        self._class_to_index = {name: idx for idx, name in enumerate(self._class_names)}

        # Gather image files
        samples: List[tuple[Path, str]] = []
        for class_dir in class_dirs:
            class_name = class_dir.name.capitalize()
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    samples.append((path, class_name))

        if not samples:
            raise RuntimeError(f"No images found under {split_dir}")

        self._samples = samples
        return samples

    def _get_valid_classes(self) -> List[str]:
        """Return list of valid class names for typographic attacks."""
        return list(self._class_names)

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        """Get sample data for a given index."""
        image_path, class_name = self._samples[index]
        image = Image.open(image_path).convert("RGB")
        class_index = self._get_class_index(class_name)
        return {"image": image, "class": class_index, "class_name": class_name}

    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        """Get sample data with class text included."""
        sample = self._get_sample_data(index)
        sample["class_text"] = self._get_class_name_from_index(sample["class"])
        return sample

    def _get_class_index(self, class_name: str) -> int:
        """Get the index of a class in the dataset."""
        return self._class_to_index[class_name]

    def _get_dataset_size(self) -> int:
        """Get the total number of samples in the dataset."""
        return len(self._samples)

    def _get_class_name_from_index(self, index: int) -> str:
        """Get the class name from a class index."""
        return self._class_names[index]


if __name__ == "__main__":
    ds = ChestXRay(root="/datasets/chest_xray_typo/", split="train")
    print(ds[0])
    import code

    code.interact(local=dict(globals(), **locals()))
