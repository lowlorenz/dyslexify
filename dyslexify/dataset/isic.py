# %%

import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
from dyslexify.dataset.base import BaseTypographicDataset


class ISIC2019Base(BaseTypographicDataset):
    """
    Base class for ISIC 2019 datasets.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        preprocess=None,
        return_index: bool = False,
        position: str = "random",
        num_workers: int = None,
    ):
        self.data_path = Path(root)

        # Initialize base class
        super().__init__(
            root=root,
            split=split,
            preprocess=preprocess,
            return_index=return_index,
            position=position,
            num_workers=num_workers,
        )

        self.templates = ["{}"]

    def _load_dataset(self, split: str) -> pd.DataFrame:
        """
        Load the ISIC 2019 dataset metadata for the specified split.

        Returns a DataFrame with label columns coerced to booleans.
        """
        data_path = self.data_path / "ISIC_2019_Training_GroundTruth.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {data_path}")

        metadata = pd.read_csv(data_path)

        # Normalize label columns to boolean for robustness
        label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
        for col in label_cols:
            if col in metadata.columns:
                values = pd.to_numeric(metadata[col], errors="coerce").fillna(0)
                metadata[col] = (values == 1.0).astype(bool)

        return metadata

    def _get_valid_classes(self) -> List[str]:
        """
        Extract valid classes from the ISIC dataset.

        Returns:
            List of valid class names that can be used for typographic attacks
        """
        return self.classes

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        row = self.dataset.iloc[index]

        train_dir = self.data_path / "Train"
        image_path = train_dir / f"{row['image']}.jpg"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)

        class_idx = self._get_class_index_from_row(row)

        return {
            "image": image,
            "class": class_idx,
            "image_name": row["image"],
        }

    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        sample_data = self._get_sample_data(index)
        class_name = self._get_class_name_from_index(sample_data["class"])
        sample_data["class_text"] = class_name
        return sample_data

    def _get_class_index(self, class_name: str) -> int:
        try:
            return self.classes.index(class_name)
        except ValueError:
            raise ValueError(f"Class '{class_name}' not found in ISIC classes")

    def _get_dataset_size(self) -> int:
        return len(self.dataset)

    def _get_class_name_from_index(self, index: int) -> str:
        if 0 <= index < len(self.classes):
            return self.classes[index]
        else:
            raise ValueError(f"Invalid class index: {index}")


class ISIC2019(ISIC2019Base):
    """
    Typographic attack dataset for ISIC skin lesion classification.

    This class extends BaseTypographicDataset to work with ISIC datasets,
    providing typographic attack functionality for skin lesion images.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        preprocess=None,
        return_index: bool = False,
        position: str = "random",
        num_workers: int = None,
    ):
        """
        Initialize the ISIC typographic dataset.

        Args:
            root: Root directory containing ISIC dataset
            split: Dataset split (train, val, test)
            preprocess: Optional preprocessing function
            return_index: Whether to return sample index
            position: Position of text on image
            num_workers: Number of worker processes for generation
        """

        self.classes = [
            "Melanoma",
            "Melanocytic nevus",
            "Basal cell carcinoma",
            "Actinic keratosis",
            "Benign keratosis",
            "Dermatofibroma",
            "Vascular lesion",
            "Squamous cell carcinoma",
            "Unknown skin lesion",
        ]

        super().__init__(
            root=root,
            split=split,
            preprocess=preprocess,
            return_index=return_index,
            position=position,
            num_workers=num_workers,
        )

    def _get_class_index_from_row(self, row: pd.Series) -> int:
        """
        Map abbreviated one-hot labels to the human-readable class index.
        Falls back to "Unknown skin lesion" when none are marked.
        """
        abbr_to_class = {
            "MEL": "Melanoma",
            "NV": "Melanocytic nevus",
            "BCC": "Basal cell carcinoma",
            "AK": "Actinic keratosis",
            "BKL": "Benign keratosis",
            "DF": "Dermatofibroma",
            "VASC": "Vascular lesion",
            "SCC": "Squamous cell carcinoma",
            "UNK": "Unknown skin lesion",
        }

        for abbr, class_name in abbr_to_class.items():
            if abbr in row and bool(row[abbr]):
                return self._get_class_index(class_name)

        return self._get_class_index("Unknown skin lesion")


class ISIC2019Binary(ISIC2019Base):
    """
    Binary typographic attack dataset for ISIC skin lesion classification.

    This class provides binary classification (melanoma vs non-melanoma)
    for typographic attacks on ISIC datasets.
    """

    def __init__(
        self,
        root: str,
        preprocess=None,
        return_index: bool = False,
        position: str = "random",
        num_workers: int = None,
    ):
        """
        Initialize the ISIC binary typographic dataset.

        Args:
            root: Root directory containing ISIC dataset
            split: Dataset split (train, val, test)
            preprocess: Optional preprocessing function
            return_index: Whether to return sample index
            position: Position of text on image
            num_workers: Number of worker processes for generation
        """
        self.classes = ["Benign", "Malignant"]
        split = "train"
        super().__init__(
            root=root,
            split=split,
            preprocess=preprocess,
            return_index=return_index,
            position=position,
            num_workers=num_workers,
        )

    def _setup_typographic_dirs(self, split: str) -> None:
        """Setup directories for typographic attack data for binary variant.
        Saves into paths with a `_binary` suffix to avoid collisions with multiclass.
        """
        self._typographic_dir = (
            self.root / f"typographic_attack_data_3fonts_{self.position}_binary" / split
        )
        self._typo_labels_path = (
            self.root / f"typographic_labels_{self.position}_{split}_binary.pt"
        )

        # Create directories if they don't exist
        self._typographic_dir.mkdir(parents=True, exist_ok=True)

    def _get_class_index_from_row(self, row: pd.Series) -> int:
        """
        Get class index from a metadata row for binary classification.

        Args:
            row: Metadata row

        Returns:
            Class index (0 or 1)
        """
        if row["MEL"] or row["BCC"] or row["SCC"] or row["AK"]:
            return 1
        else:
            return 0


if __name__ == "__main__":
    # Example usage of the multi-class ISIC dataset
    ds = ISIC2019(
        root="/datasets/isic2019_typo",
        split="train",
        position="random",
    )

    # Example usage of the binary ISIC dataset (melanoma vs non-melanoma)
    ds_binary = ISIC2019Binary(
        root="/datasets/isic2019_typo",
        split="train",
        position="random",
    )
