from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Any, Dict, Optional
from dyslexify.dataset.utils import create_typographic_attack_image


class BaseTypographicDataset(Dataset, ABC):
    """
    Abstract base class for distributed typographic dataset generation.

    This class provides a standard interface for generating typographic attack datasets
    using multiprocessing. Subclasses need to implement the specific logic for:
    - Loading and preparing the dataset
    - Extracting valid classes/texts
    - Processing individual samples
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
        Initialize the base typographic dataset.

        Args:
            root: Root directory for the dataset
            split: Dataset split (train, val, test, etc.)
            preprocess: Optional preprocessing function
            return_index: Whether to return sample index
            position: Position of text on image
            num_workers: Number of worker processes for generation
        """
        self.position = position
        self.transform = preprocess
        self.return_index = return_index
        self.num_workers = num_workers or mp.cpu_count()
        self._typo_labels = None

        # Create root directory if it doesn't exist
        self.root = Path(root)
        os.makedirs(self.root, exist_ok=True)

        # Load and prepare dataset
        self.dataset = self._load_dataset(split)

        # Setup typographic attack directories
        self._setup_typographic_dirs(split)

        # Check if we need to generate the dataset
        if not self._check_exists_synthesized_dataset():
            print("Generating typographic attack dataset...")
            self._make_typographic_attack_dataset()
        else:
            print("Loading existing typographic attack dataset...")

        # Load typo labels
        if not os.path.exists(self._typo_labels_path):
            print("Generating typographic labels...")
            self._make_typographic_attack_dataset()
        self._typo_labels = torch.load(self._typo_labels_path)

    @abstractmethod
    def _load_dataset(self, split: str) -> Any:
        """
        Load the dataset for the specified split.

        Args:
            split: Dataset split to load

        Returns:
            Dataset object (format depends on implementation)
        """
        pass

    @abstractmethod
    def _get_valid_classes(self) -> List[str]:
        """
        Extract valid classes/texts from the dataset.

        Returns:
            List of valid class names/texts that can be used for typographic attacks
        """
        pass

    @abstractmethod
    def _get_sample_data(
        self,
        index: int,
    ) -> Dict[str, Any]:
        """
        Get sample data for a given index.

        Args:
            index: Sample index

        Returns:
            Dictionary containing sample data (image, class, etc.)
        """
        pass

    @abstractmethod
    def _get_sample_with_class_text(self, index: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _get_class_index(self, class_name: str) -> int:
        """
        Get the index of a class in the dataset.

        Args:
            class_name: Name of the class

        Returns:
            Index of the class
        """
        pass

    @abstractmethod
    def _get_dataset_size(self) -> int:
        """
        Get the total number of samples in the dataset.

        Returns:
            Number of samples
        """
        pass

    @abstractmethod
    def _get_class_name_from_index(self, index: int) -> str:
        """
        Get the class name from a class index.

        Args:
            index: Class index

        Returns:
            Class name as string
        """
        pass

    def _setup_typographic_dirs(self, split: str) -> None:
        """Setup directories for typographic attack data."""
        self._typographic_dir = (
            self.root / f"typographic_attack_data_3fonts_{self.position}" / split
        )
        self._typo_labels_path = (
            self.root / f"typographic_labels_{self.position}_{split}.pt"
        )

        # Create directories if they don't exist
        os.makedirs(self._typographic_dir, exist_ok=True)

    def __len__(self) -> int:
        return self._get_dataset_size()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Get a sample from the dataset."""
        sample_data = self._get_sample_data(index)

        # Apply base transform to original image
        img = sample_data["image"]
        typo_path = self._typographic_dir / f"{index}.jpg"
        typo_img = Image.open(typo_path)

        if self.transform is not None:
            img = self.transform(img)
            typo_img = self.transform(typo_img)

        # Get labels as integer indices
        real_label = sample_data["class"]
        typo_label_idx = (
            self._typo_labels[index].item()
            if hasattr(self._typo_labels[index], "item")
            else self._typo_labels[index]
        )

        if self.return_index:
            return img, typo_img, real_label, typo_label_idx, index
        return img, typo_img, real_label, typo_label_idx

    def get_class_name(self, class_index: int) -> str:
        """
        Convert a class index to its corresponding class name.

        Args:
            class_index: Integer index of the class

        Returns:
            Class name as string
        """
        return self._get_class_name_from_index(class_index)

    def _check_exists_synthesized_dataset(self) -> bool:
        """Check if the synthesized dataset already exists."""
        if not self._typographic_dir.is_dir():
            return False

        # Check if we have all the images
        for i in range(self._get_dataset_size()):
            if not (self._typographic_dir / f"{i}.jpg").exists():
                return False

        return True

    def _make_typographic_attack_dataset(self) -> None:
        """
        Generate typographic attack dataset using multiple workers for parallel processing.
        """
        # Get valid classes
        valid_classes = self._get_valid_classes()

        # Split work among workers
        total_samples = self._get_dataset_size()
        samples_per_worker = total_samples // self.num_workers
        remainder = total_samples % self.num_workers

        # Create work chunks
        work_chunks = []
        start_idx = 0
        for i in range(self.num_workers):
            chunk_size = samples_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + chunk_size
            work_chunks.append((start_idx, end_idx))
            start_idx = end_idx

        print(f"Distributing {total_samples} samples across {self.num_workers} workers")
        print(f"Work chunks: {work_chunks}")

        # Create partial function with fixed arguments
        worker_func = partial(
            self._process_chunk,
            valid_classes=valid_classes,
            target_dir=self._typographic_dir,
            position=self.position,
        )

        # Process chunks in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(worker_func, work_chunks),
                    total=len(work_chunks),
                    desc="Processing chunks",
                )
            )

        # Combine results
        typo_labels = []
        for chunk_labels in results:
            typo_labels.extend(chunk_labels)

        # Convert to tensor and save
        typo_labels = torch.tensor(typo_labels)
        torch.save(typo_labels, self._typo_labels_path)
        print(f"Saved typographic labels to {self._typo_labels_path}")

    def _process_chunk(
        self,
        chunk: Tuple[int, int],
        valid_classes: List[str],
        target_dir: Path,
        position: str,
    ) -> List[int]:
        """
        Process a chunk of dataset indices to generate typographic attack images.

        Args:
            chunk: Tuple of (start_idx, end_idx) for this worker's work
            valid_classes: List of valid class names
            target_dir: Directory to save generated images
            position: Position of text on image

        Returns:
            List of typographic labels for this chunk
        """
        start_idx, end_idx = chunk
        typo_labels = []

        for i in range(start_idx, end_idx):
            sample_data = self._get_sample_data(i)

            text = create_typographic_attack_image(
                sample_image=sample_data["image"],
                sample_class=sample_data["class"],
                filename=f"{i}.jpg",
                dataset_classes=valid_classes,
                target_dir=target_dir,
                position=position,
            )

            typo_labels.append(self._get_class_index(text))

        return typo_labels


class BaseRealTypographicDataset(Dataset, ABC):
    """
    Abstract base class for real typographic datasets.
    """

    def __init__(self, root: str, preprocess=None):
        self.root = Path(root)
        self.preprocess = preprocess

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        pass
