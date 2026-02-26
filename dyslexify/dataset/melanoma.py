# https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download

# %%
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from dyslexify.dataset.base import BaseTypographicDataset


class Melanoma(BaseTypographicDataset):
    """
    Wraps a simple folder dataset with structure:

        root/
          train/
            benign/
              *.jpg|*.png|...
            malignant/
              *.jpg|*.png|...
          test/
            benign/
            malignant/

    into the typographic dataset pipeline defined by BaseTypographicDataset.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

    def __init__(self, *args, **kwargs):
        self.classes = ["Benign", "Malignant"]
        self.templates = ["A photo of a {} melanoma."]
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

        self._class_names = [d.name for d in class_dirs]
        self._class_to_index = {name: idx for idx, name in enumerate(self._class_names)}

        # Gather image files
        samples: List[Tuple[Path, str]] = []
        for class_dir in class_dirs:
            class_name = class_dir.name
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                    samples.append((path, class_name))

        if not samples:
            raise RuntimeError(f"No images found under {split_dir}")

        self._samples = samples
        return samples

    def _get_valid_classes(self) -> List[str]:
        return list(self._class_names)

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        image_path, class_name = self._samples[index]
        image = Image.open(image_path).convert("RGB")
        class_index = self._get_class_index(class_name)
        return {"image": image, "class": class_index, "class_name": class_name}

    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        sample = self._get_sample_data(index)
        sample["class_text"] = self._get_class_name_from_index(sample["class"])  # type: ignore[index]
        return sample

    def _get_class_index(self, class_name: str) -> int:
        if class_name not in self._class_to_index:
            raise KeyError(f"Unknown class name: {class_name}")
        return self._class_to_index[class_name]

    def _get_dataset_size(self) -> int:
        return len(self._samples)

    def _get_class_name_from_index(self, index: int) -> str:
        if index < 0 or index >= len(self._class_names):
            raise IndexError(f"Class index out of range: {index}")
        return self._class_names[index]


if __name__ == "__main__":
    ds = Melanoma(
        root="/datasets/melanoma_cancer_dataset_typo",
        split="test",
        position="bottom",
    )
    print(ds.dataset)

# %%
