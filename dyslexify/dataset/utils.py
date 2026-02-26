from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import random
import os
from pathlib import Path
from typing import Tuple, List, Union, Optional

# Type aliases for better readability
Color = str
FontName = str
Position = str

# Predefined color and font options
color: List[Color] = [
    "White",
    "Blue",
    "Green",
    "Red",
    "Magenta",
    "Cyan",
    "Yellow",
    "Black",
]

font: List[FontName] = ["Times_New_Roman.ttf", "Georgia.ttf", "Arial.ttf"]


def _transform(image: Image.Image) -> Image.Image:
    """
    Apply standard image transformations for consistent processing.

    Args:
        image: Input PIL Image to be transformed

    Returns:
        Transformed PIL Image with size 224x224, resized and center-cropped
    """
    # Pre-define transform to avoid recreating it each time
    size = 224
    transform = transforms.Compose(
        [
            transforms.Resize(
                (size, size), interpolation=Image.BICUBIC
            ),  # Size based on flag
            transforms.CenterCrop(size),
        ]
    )
    return transform(image)


def create_typographic_attack_image(
    sample_image: Image.Image,
    sample_class: str,
    filename: str,
    dataset_classes: List[str],
    target_dir: Path,
    position: Position = "random",
    font_path: str = "/usr/share/fonts/truetype/msttcorefonts/",
) -> str:
    """
    Create an image with overlaid text from a random class (different from the target class).

    This function takes an input image, adds text from a randomly selected class
    (ensuring it's different from the target class at index idx), and saves the
    result to the target directory.

    Args:
        file: Path to the input image file
        classes: List of available text classes to choose from
        img_dir: Directory containing the input images
        target_dir: Directory where the processed images will be saved
        idx: Index of the target class (text will be chosen from other classes)
        position: Position of text on image ("random", "top", "bottom", "left", "right", "center")
        font_path: Path to the directory containing font files

    Returns:
        The randomly chosen text class that was overlaid on the image

    Raises:
        FileNotFoundError: If the input image file doesn't exist
        OSError: If there are issues with file operations or font loading
    """
    img = _transform(sample_image)
    text = random.choice(dataset_classes)
    font_path = os.path.join(font_path, random.choice(font))
    while text == sample_class:
        text = random.choice(dataset_classes)
    fill, stroke = random.choice(color), random.choice(color)
    while fill == stroke:
        stroke = random.choice(color)

    img = create_image(img, text, font_path, fill, stroke, position)
    dir = target_dir / "/".join(str(filename).split("/")[:-1])

    os.makedirs(dir, exist_ok=True)
    img.save(target_dir / filename, quality=100)

    # Return the chosen text class
    return text


def create_image(
    image: Image.Image,
    text: str,
    font_path: str,
    fill: Color,
    stroke: Color,
    position: Position = "random",
) -> Image.Image:
    """
    Overlay text on an image with specified styling and position.

    Args:
        image: Input PIL Image to overlay text on
        text: Text string to overlay on the image
        font_path: Path to the font file to use
        fill: Color for the text fill
        stroke: Color for the text stroke/outline
        position: Position of text on image ("random", "top", "bottom", "left", "right", "center")

    Returns:
        PIL Image with text overlaid

    Raises:
        OSError: If the font file cannot be loaded
    """

    image = _transform(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    W, H = image.size
    draw = ImageDraw.Draw(image)

    # Ensure text is single line
    text = " ".join(text.split())

    font, txpos, _ = adjust_font_size((W, H), draw, text, font_path, position)
    draw.text(txpos, text, font=font, fill=fill, stroke_fill=stroke, stroke_width=1)
    return image


def adjust_font_size(
    img_size: Tuple[int, int],
    imagedraw: ImageDraw.Draw,
    text: str,
    font_path: str,
    position: Position = "random",
) -> Tuple[ImageFont.FreeTypeFont, Tuple[float, float], Tuple[int, int]]:
    """
    Calculate appropriate font size and position for text to fit within image bounds.

    Uses binary search to efficiently find the largest font size that allows
    the text to fit within the image dimensions.

    Args:
        img_size: Tuple of (width, height) of the image
        imagedraw: PIL ImageDraw object for text measurement
        text: Text string to measure and position
        font_path: Path to the font file
        position: Position of text on image ("random", "top", "bottom", "left", "right", "center")

    Returns:
        Tuple containing:
        - PIL ImageFont object with appropriate size
        - Tuple of (x, y) coordinates for text position
        - Tuple of (width, height) of the text bounding box

    Raises:
        OSError: If the font file cannot be loaded
    """
    W, H = img_size
    # Start with a more reasonable font size to reduce iterations
    font_size = min(30, H // 6)  # Start with estimated size
    font = ImageFont.truetype(font_path, font_size)

    bbox = imagedraw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # Binary search instead of linear reduction
    while w >= W or h >= H:
        font_size = font_size // 2
        font = ImageFont.truetype(font_path, font_size)
        bbox = imagedraw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

    # Calculate position based on specified location
    if position == "random":
        txpos = ((W - w) * random.random(), (H - h) * random.random())
    elif position == "top":
        txpos = ((W - w) / 2, h / 2)
    elif position == "bottom":
        txpos = ((W - w) / 2, H - h * 1.5)
    elif position == "left":
        txpos = (w / 2, (H - h) / 2)
    elif position == "right":
        txpos = (W - w * 1.5, (H - h) / 2)
    elif position == "center":
        txpos = ((W - w) / 2, (H - h) / 2)

    return font, txpos, (w, h)
