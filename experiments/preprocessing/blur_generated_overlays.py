import cv2
import numpy as np
from pathlib import Path
import easyocr
from PIL import Image
from tqdm import tqdm
import json
import argparse


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
    parser = argparse.ArgumentParser(
        description="Blur text in generated typographic overlay images"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/datasets/imagenet-100-typo-blur",
        help="Root directory of the dataset",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="random",
        help="Position mode (e.g., random)",
    )
    parser.add_argument(
        "--split", type=str, default="val", help="Dataset split (e.g., val)"
    )
    parser.add_argument(
        "--kernel-size", type=int, default=51, help="Gaussian blur kernel size"
    )
    parser.add_argument(
        "--sigma", type=int, default=25, help="Gaussian blur sigma"
    )

    args = parser.parse_args()

    root = Path(args.root)
    position = args.position
    split = args.split

    # Path to generated typographic overlays
    source_dir = (
        root / f"typographic_attack_data_3fonts_{position}" / split
    )
    
    # Output directory for blurred images
    output_dir = (
        root / f"typographic_attack_data_3fonts_{position}_blurred" / split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")

    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        print(
            "Please first run the dataset to generate typographic overlays, then run this script."
        )
        return

    # Get all images
    image_files = sorted(list(source_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print("No images found! Make sure the typographic overlays have been generated.")
        return

    # Initialize EasyOCR
    print("Initializing EasyOCR...")
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    # Process all images
    for image_path in tqdm(image_files, desc="Blurring text in images"):
        output_path = output_dir / image_path.name

        try:
            blurred_img = process_image(
                image_path, reader, kernel_size=args.kernel_size, sigma=args.sigma
            )
            cv2.imwrite(str(output_path), blurred_img)
        except Exception as e:
            print(f"\nError processing {image_path}: {e}")
            # Copy original if blurring fails
            import shutil

            shutil.copy2(image_path, output_path)

    # Create metadata file
    metadata = {
        "blurred_dir": str(output_dir),
        "source_dir": str(source_dir),
        "position": position,
        "split": split,
        "kernel_size": args.kernel_size,
        "sigma": args.sigma,
        "num_images": len(image_files),
        "status": "complete",
    }

    metadata_path = root / f"blurred_metadata_{position}_{split}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Processing complete!")
    print(f"  Blurred images saved to: {output_dir}")
    print(f"  Metadata saved to: {metadata_path}")
    print(f"\nThe dataset will now automatically use these blurred images.")


if __name__ == "__main__":
    main()


