import json
import torch
import torch.nn
import open_clip
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

from dyslexify.config import MODELS, DEVICE
from dyslexify.dataset import HAM10k
from dyslexify.zeroshot import calculate_text_features
from dyslexify.defend import dislexify_openclip_model
from dyslexify.cache.collector import change_attn_implementation_to_hookable


def load_models(device: str) -> Tuple:
    """Load original and defended WhyLesion CLIP models."""
    model_name = MODELS["whylesionclip"]["model_name"]
    pretrained = MODELS["whylesionclip"]["pretrained"]

    # Load original model
    original_model, _, transform = open_clip.create_model_and_transforms(
        model_name, pretrained
    )
    original_model.to(device)

    # Load defended model
    defended_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
    defended_model.to(device)

    # Apply defense to the second model
    typographic_attention_heads_path = (
        "results/experiments/greedy_selection/whylesionclip/ablated_heads_cls.json"
    )
    with open(typographic_attention_heads_path, "r") as f:
        typographic_attention_heads = json.load(f)

    change_attn_implementation_to_hookable(defended_model)
    defended_model = dislexify_openclip_model(
        defended_model,
        typographic_attention_heads=typographic_attention_heads,
        mode="cls",
    )

    tokenizer = open_clip.get_tokenizer(model_name)

    return original_model, defended_model, transform, tokenizer


def prediction_logits_with_scaling(
    model: torch.nn.Module,
    images: torch.Tensor,
    text_features: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute prediction logits with proper logit scaling coefficient."""
    img_features = model.encode_image(images)
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    # Apply logit scaling - OpenCLIP models have a learnable logit_scale parameter
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (img_features @ text_features.T)
    preds = logits.argmax(dim=1)
    return logits, preds


def find_melanocytic_nevi_misclassifications(
    original_model, defended_model, dataset, dataloader, text_features, device: str
) -> List[Dict]:
    """Find samples where typo label is melanocytic nevi, original model fails, defended model succeeds."""
    results = []
    melanocytic_nevi_class_idx = dataset._get_class_index("Melanocytic Nevi")

    print(f"Looking for melanocytic nevi samples (class index: {melanocytic_nevi_class_idx})...")

    with torch.inference_mode():
        for batch_idx, (
            original_image,
            typographic_image,
            target,
            typo_label,
        ) in enumerate(dataloader):
            original_image = original_image.to(device)
            typographic_image = typographic_image.to(device)
            target = target.to(device)
            typo_label = typo_label.to(device)

            # Filter for melanocytic nevi typo labels
            melanocytic_nevi_mask = typo_label == melanocytic_nevi_class_idx
            if not melanocytic_nevi_mask.any():
                continue

            # Get predictions from both models on typographic images
            original_logits, original_preds = prediction_logits_with_scaling(
                original_model, typographic_image, text_features
            )
            defended_logits, defended_preds = prediction_logits_with_scaling(
                defended_model, typographic_image, text_features
            )

            # Find samples where:
            # 1. Typo label is melanocytic nevi
            # 2. Original model misclassifies
            # 3. Defended model correctly classifies
            for i in range(len(target)):
                if not melanocytic_nevi_mask[i]:
                    continue

                sample_idx = batch_idx * dataloader.batch_size + i
                true_label = target[i].item()
                typo_true_label = typo_label[i].item()

                original_pred = original_preds[i].item()
                defended_pred = defended_preds[i].item()

                # Check if original model misclassifies and defended model gets it right
                original_correct = original_pred == true_label
                defended_correct = defended_pred == true_label

                if not original_correct and defended_correct:
                    # Get the image path for this sample
                    row = dataset.dataset.iloc[sample_idx]
                    image_id = row["image_id"]
                    image_path = f"/datasets/HAM/typographic_attack_data_3fonts_random/train/{sample_idx}.jpg"
                    original_image_path = f"/datasets/HAM/images/{image_id}.jpg"

                    # Convert logits to probabilities
                    original_probs = torch.softmax(original_logits[i], dim=0)
                    defended_probs = torch.softmax(defended_logits[i], dim=0)

                    sample_result = {
                        "image_path": image_path,
                        "original_image_path": original_image_path,
                        "image_id": image_id,
                        "true_class": dataset.classes[true_label],
                        "true_class_index": true_label,
                        "typo_class": dataset.classes[typo_true_label],
                        "typo_class_index": typo_true_label,
                        "original_prediction": dataset.classes[original_pred],
                        "original_prediction_index": original_pred,
                        "defended_prediction": dataset.classes[defended_pred],
                        "defended_prediction_index": defended_pred,
                        "original_probabilities": {
                            dataset.classes[j]: original_probs[j].item()
                            for j in range(len(dataset.classes))
                        },
                        "defended_probabilities": {
                            dataset.classes[j]: defended_probs[j].item()
                            for j in range(len(dataset.classes))
                        },
                        "original_confidence": original_probs[original_pred].item(),
                        "defended_confidence": defended_probs[defended_pred].item(),
                        "true_class_prob_original": original_probs[true_label].item(),
                        "true_class_prob_defended": defended_probs[true_label].item(),
                    }

                    results.append(sample_result)
                    print(
                        f"Found sample {image_id}: "
                        f"True={dataset.classes[true_label]}, "
                        f"Typo={dataset.classes[typo_true_label]}, "
                        f"Original={dataset.classes[original_pred]}, "
                        f"Defended={dataset.classes[defended_pred]}"
                    )

    return results


def main():
    device = DEVICE
    print(f"Using device: {device}")

    # Load models
    print("Loading WhyLesion CLIP models...")
    original_model, defended_model, transform, tokenizer = load_models(device)

    # Load HAM10k dataset
    print("Loading HAM10k dataset...")
    dataset = HAM10k(
        root="/datasets/HAM", split="train", preprocess=transform, position="random"
    )

    # Create dataloader - using smaller batch size for memory efficiency
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    print(f"Dataset loaded with {len(dataset)} samples")
    print(f"Classes: {dataset.classes}")

    # Calculate text features
    print("Calculating text features...")
    text_features = calculate_text_features(original_model, dataset, tokenizer, device)

    # Find misclassifications
    print("Finding melanocytic nevi misclassifications...")
    results = find_melanocytic_nevi_misclassifications(
        original_model, defended_model, dataset, dataloader, text_features, device
    )

    # Save results
    output_file = "results/plots/melanocytic_nevi_misclassification_analysis.json"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    final_results = {
        "metadata": {
            "dataset": "HAM10k",
            "model": "whylesionclip",
            "total_samples": len(dataset),
            "correctly_classified_samples_found": len(results),
            "filter_criteria": {
                "typo_label": "Melanocytic Nevi",
                "original_model_incorrect": True,
                "defended_model_correct": True,
                "note": "Only samples correctly classified by defended model are included",
            },
            "typographic_attention_heads": "results/experiments/greedy_selection/whylesionclip/ablated_heads_cls.json",
            "data_fields": {
                "image_path": "Path to typographic attack image",
                "original_image_path": "Path to original HAM image",
                "image_id": "HAM dataset image ID",
            },
        },
        "samples": results,
    }

    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nAnalysis complete!")
    print(f"Found {len(results)} samples where:")
    print(f"- Typo label is 'Melanocytic Nevi'")
    print(f"- Original WhyLesion CLIP misclassified")
    print(f"- Defended WhyLesion CLIP correctly classified")
    print(f"- Image paths included for both original and typographic versions")
    print(f"\nResults saved to: {output_path}")

    # Print summary statistics
    if results:
        print(f"\nSummary Statistics:")
        true_classes = [r["true_class"] for r in results]
        original_preds = [r["original_prediction"] for r in results]

        print(f"True class distribution:")
        for cls in set(true_classes):
            count = true_classes.count(cls)
            print(f"  {cls}: {count}")

        print(f"Original model's wrong predictions:")
        for cls in set(original_preds):
            count = original_preds.count(cls)
            print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
