# %%

import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
from dyslexify.dataset.base import BaseTypographicDataset


class HAM10kBase(BaseTypographicDataset):

    """
    Base class for HAM10k datasets.
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
        Load the HAM10k dataset metadata for the specified split.

        Returns a DataFrame with the HAM10k data.
        """
        data_path = self.data_path / "HAM10000_metadata.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"HAM10k data file not found: {data_path}")

        metadata = pd.read_csv(data_path)

        # HAM10k doesn't have explicit splits, so we'll use all data for any split
        # In practice, you might want to implement a proper train/test split here

        return metadata

    def _get_valid_classes(self) -> List[str]:
        """
        Extract valid classes from the HAM10k dataset.

        Returns:
            List of valid class names that can be used for typographic attacks
        """
        return self.classes

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        row = self.dataset.iloc[index]

        # HAM10k images are stored with their image_id as filename
        image_id = row["image_id"]
        image_filename = f"{image_id}.jpg"

        # Images are in the images subdirectory
        image_path = self.data_path / "images" / image_filename


        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)

        class_idx = self._get_class_index_from_row(row)

        return {
            "image": image,
            "class": class_idx,
            "image_name": image_id,
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
            raise ValueError(f"Class '{class_name}' not found in HAM10k classes")

    def _get_dataset_size(self) -> int:
        return len(self.dataset)

    def _get_class_name_from_index(self, index: int) -> str:
        if 0 <= index < len(self.classes):
            return self.classes[index]
        else:
            raise ValueError(f"Invalid class index: {index}")


class HAM10k(HAM10kBase):
    """
    Typographic attack dataset for HAM10k skin lesion classification.

    This class extends BaseTypographicDataset to work with HAM10k datasets,
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
        Initialize the HAM10k typographic dataset.

        Args:
            root: Root directory containing HAM10k dataset
            split: Dataset split (train, val, test)
            preprocess: Optional preprocessing function
            return_index: Whether to return sample index
            position: Position of text on image
            num_workers: Number of worker processes for generation
        """

        self.classes = [
            "Actinic Keratoses",
            "Basal Cell Carcinoma",
            "Benign Keratosis-like Lesions",
            "Dermatofibroma",
            "Melanoma",
            "Melanocytic Nevi",
            "Vascular Lesions",
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
        Map HAM10k diagnosis codes to the class names.
        """
        diagnosis_to_class = {
            "akiec": "Actinic Keratoses",
            "bcc": "Basal Cell Carcinoma",
            "bkl": "Benign Keratosis-like Lesions",
            "df": "Dermatofibroma",
            "mel": "Melanoma",
            "nv": "Melanocytic Nevi",
            "vasc": "Vascular Lesions",
        }

        diagnosis = row["dx"]

        if diagnosis in diagnosis_to_class:
            class_name = diagnosis_to_class[diagnosis]
            return self._get_class_index(class_name)
        else:
            raise ValueError(f"Unsupported diagnosis: {diagnosis}")


class HAM10kBinary(HAM10kBase):
    """
    Binary typographic attack dataset for HAM10k skin lesion classification.

    This class provides binary classification (malignant vs benign)
    for typographic attacks on HAM10k datasets.
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
        Initialize the HAM10k binary typographic dataset.

        Args:
            root: Root directory containing HAM10k dataset
            split: Dataset split (train, val, test)
            preprocess: Optional preprocessing function
            return_index: Whether to return sample index
            position: Position of text on image
            num_workers: Number of worker processes for generation
        """
        self.classes = ["Benign", "Malignant"]

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
            Class index (0 for benign, 1 for malignant)
        """
        diagnosis = row["dx"]

        # Malignant diagnoses in HAM10k
        malignant_diagnoses = ["mel", "bcc", "akiec"]

        if diagnosis in malignant_diagnoses:
            return 1  # Malignant
        else:
            return 0  # Benign


if __name__ == "__main__":
    # Example usage of the multi-class HAM10k dataset
    ds = HAM10k(
        root="/data/datasets/HAM",
        split="train",
        position="random",
    )

    # Example usage of the binary HAM10k dataset (malignant vs benign)
    ds_binary = HAM10kBinary(
        root="/data/datasets/HAM",
        split="train",
        position="random",
    )
# %%
