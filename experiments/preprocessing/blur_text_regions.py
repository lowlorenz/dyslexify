import cv2
import numpy as np
from pathlib import Path
import easyocr
from PIL import Image
from tqdm import tqdm
import shutil
import json


def apply_gaussian_blur_to_region(
    image: np.ndarray, bbox: list, kernel_size: int = 51, sigma: int = 25
) -> np.ndarray:
    """Apply Gaussian blur to a specific region defined by bounding box."""
    points = np.array(bbox, dtype=np.int32)
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min, x_max = max(0, x_coords.min()), min(image.shape[1], x_coords.max())
    y_min, y_max = max(0, y_coords.min()), min(image.shape[0], y_coords.max())

    if x_max > x_min and y_max > y_min:
        roi = image[y_min:y_max, x_min:x_max].copy()
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), sigma)
        image[y_min:y_max, x_min:x_max] = blurred_roi

    return image


def process_image(
    image_path: Path, reader, kernel_size: int = 51, sigma: int = 25
) -> np.ndarray:
    """Process a single image: detect text and blur regions."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    results = reader.readtext(str(image_path))

    if not results:
        return img

    for bbox, text, prob in results:
        img = apply_gaussian_blur_to_region(img, bbox, kernel_size, sigma)

    return img


def main():
    source_root = Path("/datasets/imagenet-100-typo")
    target_root = Path("/datasets/imagenet-100-typo-blur")

    source_val_dir = source_root / "data" / "val"
    target_val_dir = target_root / "data" / "val"

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    print(f"Source directory: {source_val_dir}")
    print(f"Target directory: {target_val_dir}")

    image_files = (
        list(source_val_dir.rglob("*.JPEG"))
        + list(source_val_dir.rglob("*.jpg"))
        + list(source_val_dir.rglob("*.png"))
    )
    print(f"Found {len(image_files)} images to process")

    target_val_dir.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(image_files, desc="Processing images"):
        relative_path = image_path.relative_to(source_val_dir)
        target_path = target_val_dir / relative_path

        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            blurred_img = process_image(image_path, reader, kernel_size=51, sigma=25)
            cv2.imwrite(str(target_path), blurred_img)
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            shutil.copy2(image_path, target_path)

    labels_source = source_root / "data" / "Labels.json"
    labels_target = target_root / "data" / "Labels.json"

    if labels_source.exists():
        labels_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(labels_source, labels_target)
        print(f"\nCopied Labels.json to {labels_target}")

    print("\nProcessing complete!")
    print(f"Blurred images saved to: {target_val_dir}")


if __name__ == "__main__":
    main()
