from pathlib import Path
import json
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple
from dyslexify.dataset.utils import _transform
from dyslexify.dataset.base import BaseTypographicDataset
from dyslexify.dataset.ImageNet100 import imagenet_100_classes, imagenet_100_templates


class ImageNet100Blurred(BaseTypographicDataset):
    def __init__(
        self,
        root,
        split="val",
        preprocess=None,
        return_index=False,
        position="random",
        num_workers=None,
    ):
        with open(Path(root) / "data" / "Labels.json") as f:
            self.id_to_class = json.load(f)

        self._data_dir = Path(root) / "data" / split

        self._load_files_and_labels()

        self.templates = imagenet_100_templates

        # Check if blurred images exist
        self.blurred_meta_path = (
            Path(root) / f"blurred_metadata_{position}_{split}.json"
        )
        self.use_blurred = self.blurred_meta_path.exists()

        if self.use_blurred:
            with open(self.blurred_meta_path, "r") as f:
                self.blurred_metadata = json.load(f)
            print(
                f"Loading pre-blurred images from {self.blurred_metadata['blurred_dir']}"
            )

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
        classes = []
        for dir in self._data_dir.iterdir():
            if not dir.is_dir():
                continue
            id = str(dir).split("/")[-1]
            class_i = self.id_to_class[id].split(",")[0]
            classes.append(class_i)

        self.classes = classes
        self.class_to_idx = dict(zip(classes, range(len(classes))))

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
        """Load the ImageNet100Blurred dataset for the specified split."""
        return self._files

    def _get_valid_classes(self) -> List[str]:
        """Extract valid classes from the ImageNet100Blurred dataset."""
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

    def __getitem__(self, index: int) -> Tuple:
        """Override to load pre-blurred images if available."""
        if self.use_blurred:
            # Load pre-blurred typographic overlay
            blurred_dir = Path(self.blurred_metadata["blurred_dir"])
            img_filename = f"{index}.jpg"
            blurred_path = blurred_dir / img_filename

            if blurred_path.exists():
                # Load blurred overlay
                typo_img = Image.open(blurred_path)
                if self.transform:
                    typo_img = self.transform(_transform(typo_img))
                else:
                    typo_img = _transform(typo_img)

                # Load original image (without text)
                img_path = self._files[index]
                img = Image.open(img_path)
                if self.transform:
                    img = self.transform(_transform(img))
                else:
                    img = _transform(img)

                # Get labels
                real_label = self._labels[index]
                pos = self.blurred_metadata["position"]
                spl = self.blurred_metadata["split"]
                typo_label_file = (
                    blurred_dir.parent.parent / f"typographic_labels_{pos}_{spl}.pt"
                )
                if typo_label_file.exists():
                    typo_labels = torch.load(typo_label_file)
                    typo_label = typo_labels[index].item()
                else:
                    typo_label = real_label

                if self.return_index:
                    return img, typo_img, real_label, typo_label, index
                return img, typo_img, real_label, typo_label

        # Fall back to parent class generation
        return super().__getitem__(index)
