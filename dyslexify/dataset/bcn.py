# %%

import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List
from dyslexify.dataset.base import BaseTypographicDataset


class BCN20kBase(BaseTypographicDataset):
    """
    Base class for BCN 20k datasets.
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
        Load the BCN 20k dataset metadata for the specified split.

        Returns a DataFrame with the BCN data.
        """
        if split == "train":
            data_path = self.data_path / "bcn_20k_train.csv"
        else:
            # For test/val, we'll use train data (since only train CSV is provided)
            # In practice, you might want to split the train data
            data_path = self.data_path / "bcn_20k_train.csv"

        if not data_path.exists():
            raise FileNotFoundError(f"BCN data file not found: {data_path}")

        metadata = pd.read_csv(data_path)

        # Filter by split if column exists
        if "split" in metadata.columns and split != "train":
            # If we want test data but only have train CSV, we'll use all data
            # You might want to implement a proper train/test split here
            pass

        return metadata

    def _get_valid_classes(self) -> List[str]:
        """
        Extract valid classes from the BCN dataset.

        Returns:
            List of valid class names that can be used for typographic attacks
        """
        return self.classes

    def _get_sample_data(self, index: int) -> Dict[str, Any]:
        row = self.dataset.iloc[index]

        # Determine the correct image directory based on the filename
        # BCN test images are in BCN_20k_test/bcn_20k_test/
        # Training images should be in a similar structure or main directory
        bcn_filename = row["bcn_filename"]

        # Try test directory first
        test_dir = self.data_path / "BCN_20k_test" / "bcn_20k_test"
        image_path = test_dir / bcn_filename

        # If not found in test, try main directory or train directory
        if not image_path.exists():
            # Try other possible locations
            train_dir = self.data_path / "BCN_20k_train" / "bcn_20k_train"
            if train_dir.exists():
                image_path = train_dir / bcn_filename
            else:
                # Try main directory
                image_path = self.data_path / bcn_filename

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path)

        class_idx = self._get_class_index_from_row(row)

        return {
            "image": image,
            "class": class_idx,
            "image_name": bcn_filename.replace(".jpg", ""),
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
            raise ValueError(f"Class '{class_name}' not found in BCN classes")

    def _get_dataset_size(self) -> int:
        return len(self.dataset)

    def _get_class_name_from_index(self, index: int) -> str:
        if 0 <= index < len(self.classes):
            return self.classes[index]
        else:
            raise ValueError(f"Invalid class index: {index}")


class BCN20k(BCN20kBase):
    """
    Typographic attack dataset for BCN 20k skin lesion classification.

    This class extends BaseTypographicDataset to work with BCN datasets,
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
        Initialize the BCN typographic dataset.

        Args:
            root: Root directory containing BCN dataset
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
            "Melanocytic Nevi",
            "Melanoma",
            "Squamous Cell Carcinoma",
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
        Map BCN diagnosis codes to the class names.
        """
        diagnosis_to_class = {
            "AK": "Actinic Keratoses",
            "BCC": "Basal Cell Carcinoma",
            "BKL": "Benign Keratosis-like Lesions",
            "DF": "Dermatofibroma",
            "NV": "Melanocytic Nevi",
            "MEL": "Melanoma",
            "SCC": "Squamous Cell Carcinoma",
            "VASC": "Vascular Lesions",
        }

        diagnosis = row["diagnosis"]

        if diagnosis in diagnosis_to_class:
            class_name = diagnosis_to_class[diagnosis]
            return self._get_class_index(class_name)
        else:
            raise ValueError(f"Unsupported diagnosis: {diagnosis}")


class BCN20kBinary(BCN20kBase):
    """
    Binary typographic attack dataset for BCN 20k skin lesion classification.

    This class provides binary classification (malignant vs benign)
    for typographic attacks on BCN datasets.
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
        Initialize the BCN binary typographic dataset.

        Args:
            root: Root directory containing BCN dataset
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
        diagnosis = row["diagnosis"]

        # Malignant diagnoses
        malignant_diagnoses = ["MEL", "BCC", "SCC"]

        if diagnosis in malignant_diagnoses:
            return 1  # Malignant
        else:
            return 0  # Benign


if __name__ == "__main__":
    # Example usage of the multi-class BCN dataset
    ds = BCN20k(
        root="/datasets/BCN_20k",
        split="train",
        position="random",
    )

    # Example usage of the binary BCN dataset (malignant vs benign)
    ds_binary = BCN20kBinary(
        root="/datasets/BCN_20k",
        split="train",
        position="random",
    )
# %%
