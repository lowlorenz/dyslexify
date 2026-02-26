from typing import Callable, List, Optional, Tuple, Any, Dict
from torch import nn
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import einops
from dyslexify.dataset.base import BaseTypographicDataset


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def zeroshot_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    device: str,
    tqdm_active: bool = True,
) -> Tuple[float, float]:
    """Evaluate model performance on original and typographic images using zero-shot classification.

    This function performs zero-shot classification on a dataset containing both original
    and typographic (dyslexic-friendly) versions of images. It computes accuracy metrics
    for both image types to assess the model's robustness to typographic changes.

    Args:
        model (nn.Module): The neural network model to evaluate. Should be a vision model
            that can process images and output feature representations.
        dataloader (DataLoader): DataLoader containing batches of (original_image,
            typographic_image, target, typo_label) tuples. Each batch should contain:
            - original_image: Original version of the images
            - typographic_image: Dyslexic-friendly typographic version of the images
            - target: Ground truth labels for classification
            - typo_label: Labels for the typographic images (typically same as target)
        text_features (torch.Tensor): Pre-computed text feature embeddings for all
            possible class labels. Shape should be (num_classes, feature_dim).
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        Tuple[float, float]: A tuple containing:
            - regular_accuracy: Classification accuracy on original images
            - typo_accuracy: Classification accuracy on typographic images

    Example:
        >>> model = load_pretrained_model()
        >>> text_features = compute_text_features(class_names)
        >>> regular_acc, typo_acc = zeroshot_classifier(
        ...     model, dataloader, text_features, device='cuda'
        ... )
        >>> print(f"Original accuracy: {regular_acc:.3f}")
        >>> print(f"Typographic accuracy: {typo_acc:.3f}")
    """
    regular_correct = 0
    typo_correct = 0
    total = 0

    for original_image, typographic_image, target, typo_label in tqdm(
        dataloader,
        desc=f"Zero-shot classification on {dataloader.dataset.__class__.__name__}",
        unit_scale=dataloader.batch_size,
        unit="images",
        disable=not tqdm_active,
    ):
        original_image = original_image.to(device)
        typographic_image = typographic_image.to(device)
        target = target.to(device)
        typo_label = typo_label.to(device)

        with torch.no_grad():
            original_logits, original_preds = prediction_logits(
                model, original_image, text_features
            )
            typographic_logits, typographic_preds = prediction_logits(
                model, typographic_image, text_features
            )

            regular_correct += (original_preds == target).sum().item()
            typo_correct += (typographic_preds == target).sum().item()
            total += len(target)

    return regular_correct / total, typo_correct / total


def prediction_logits(
    model: nn.Module,
    images: torch.Tensor,
    text_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute prediction logits and class predictions using cosine similarity.

    This function performs zero-shot classification by computing cosine similarity
    between image features and text features. The image features are first normalized,
    then compared against the normalized text features to compute logits and predictions.

    Args:
        model (nn.Module): The neural network model to extract image features from.
            Should output feature representations that can be compared with text features.
        img_features (torch.Tensor): Image feature embeddings from the model.
            Shape should be (batch_size, feature_dim).
        text_features (torch.Tensor): Text feature embeddings for all possible classes.
            Shape should be (num_classes, feature_dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - logits: Cosine similarity scores between image and text features.
                Shape: (batch_size, num_classes)
            - preds: Predicted class indices (argmax of logits).
                Shape: (batch_size,)

    Example:
        >>> img_features = model(images)  # Shape: (32, 512)
        >>> text_features = text_encoder(class_names)  # Shape: (100, 512)
        >>> logits, predictions = prediction_logits(model, img_features, text_features)
        >>> print(f"Logits shape: {logits.shape}")  # (32, 100)
        >>> print(f"Predictions shape: {predictions.shape}")  # (32,)
    """
    img_features = model.encode_image(images)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    logits = img_features @ text_features.T
    preds = logits.argmax(dim=1)
    return logits, preds


def calculate_text_features(
    model: nn.Module,
    dataset: BaseTypographicDataset,
    tokenizer: Any,
    device: str,
) -> torch.Tensor:
    """
    Calculate text features by averaging each sample over all templates.

    This method computes text embeddings for all class names using multiple
    templates and averages them to create robust text features for zero-shot
    classification.

    Args:
        tokenizer: Tokenizer for processing text inputs

    Returns:
        torch.Tensor: Text features tensor of shape (num_classes, feature_dim)
    """
    text_features = []

    for class_name in dataset.classes:
        # build batch over all templates
        batch = []
        for template in dataset.templates:
            tokens = tokenizer(template.format(class_name))
            batch.append(tokens)

        batch = torch.cat(batch, dim=0).to(device)
        batch_features = model.encode_text(batch)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        batch_features = einops.reduce(batch_features, "batch dim -> dim", "mean")
        text_features.append(batch_features)

    text_features = torch.stack(text_features).to(device)

    return text_features


@torch.inference_mode()
@torch.autocast("cuda", dtype=torch.bfloat16)
def zeroshot_retrieval(
    model: nn.Module,
    dataloader: DataLoader,
    text_features: torch.Tensor,
    device: str,
    top_k: int = 10,
    tqdm_active: bool = True,
) -> Tuple[
    Dict[int, List[Tuple[int, float]]],
    Dict[int, List[Tuple[int, float]]],
    Dict[int, List[Tuple[int, float]]],
]:
    """Retrieve top-k images per class based on cosine similarity scores.

    This function performs zero-shot retrieval by computing cosine similarity between
    image features and text features for each class. It returns the top-k most similar
    images for each class in three configurations: regular images only, typographic
    images only, and mixed (both types combined).

    Args:
        model (nn.Module): The neural network model to evaluate. Should be a vision model
            that can process images and output feature representations.
        dataloader (DataLoader): DataLoader containing batches of (original_image,
            typographic_image, target, typo_label) tuples. The dataset should have
            return_index=True to provide image indices.
        text_features (torch.Tensor): Pre-computed text feature embeddings for all
            possible class labels. Shape should be (num_classes, feature_dim).
        device (str): Device to run the model on ('cpu' or 'cuda').
        top_k (int): Number of top images to retrieve per class. Defaults to 10.
        tqdm_active (bool): Whether to show progress bar. Defaults to True.

    Returns:
        Tuple[Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]], Dict[int, List[Tuple[int, float]]]]:
            A tuple containing three dictionaries:
            - regular_top_k: Top-k regular images per class {class_id: [(img_id, similarity), ...]}
            - typo_top_k: Top-k typographic images per class {class_id: [(img_id, similarity), ...]}
            - mixed_top_k: Top-k mixed images per class {class_id: [(img_id, similarity), ...]}

    Example:
        >>> model = load_pretrained_model()
        >>> text_features = compute_text_features(class_names)
        >>> regular_top, typo_top, mixed_top = zeroshot_retrieval(
        ...     model, dataloader, text_features, device='cuda', top_k=5
        ... )
        >>> print(f"Top 5 regular images for class 0: {regular_top[0]}")
        >>> print(f"Top 5 typo images for class 0: {typo_top[0]}")
    """
    num_classes = text_features.shape[0]

    # Storage for similarities and image IDs
    regular_similarities = {class_idx: [] for class_idx in range(num_classes)}
    typo_similarities = {class_idx: [] for class_idx in range(num_classes)}
    mixed_similarities = {class_idx: [] for class_idx in range(num_classes)}

    batch_idx = 0
    for batch_data in tqdm(
        dataloader,
        desc=f"Zero-shot retrieval on {dataloader.dataset.__class__.__name__}",
        unit_scale=dataloader.batch_size,
        unit="images",
        disable=not tqdm_active,
    ):

        original_image, typographic_image, target, typo_label = batch_data
        indices = torch.arange(
            batch_idx * dataloader.batch_size,
            batch_idx * dataloader.batch_size + len(target),
        )

        original_image = original_image.to(device)
        typographic_image = typographic_image.to(device)
        target = target.to(device)
        typo_label = typo_label.to(device)

        with torch.no_grad():
            original_logits, _ = prediction_logits(model, original_image, text_features)
            typographic_logits, _ = prediction_logits(
                model, typographic_image, text_features
            )

            for i in range(len(target)):
                img_idx = (
                    indices[i].item() if hasattr(indices[i], "item") else indices[i]
                )
                for class_idx in range(num_classes):
                    reg_sim = original_logits[i, class_idx].item()
                    typo_sim = typographic_logits[i, class_idx].item()

                    regular_similarities[class_idx].append((img_idx, reg_sim))
                    typo_similarities[class_idx].append((img_idx, typo_sim))

                    mixed_similarities[class_idx].append((img_idx, reg_sim))
                    mixed_similarities[class_idx].append((-img_idx - 1, typo_sim))

        batch_idx += 1

    # Get top-k for each class
    regular_top_k = {}
    typo_top_k = {}
    mixed_top_k = {}

    for class_idx in range(num_classes):
        # Sort by similarity (descending) and take top-k
        regular_sorted = sorted(
            regular_similarities[class_idx], key=lambda x: x[1], reverse=True
        )
        typo_sorted = sorted(
            typo_similarities[class_idx], key=lambda x: x[1], reverse=True
        )
        mixed_sorted = sorted(
            mixed_similarities[class_idx], key=lambda x: x[1], reverse=True
        )

        regular_top_k[class_idx] = regular_sorted[:top_k]
        typo_top_k[class_idx] = typo_sorted[:top_k]
        mixed_top_k[class_idx] = mixed_sorted[:top_k]

    return regular_top_k, typo_top_k, mixed_top_k
