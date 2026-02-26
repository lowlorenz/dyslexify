import json
from pathlib import Path
from typing import Dict, Any


def load_results(model_short_name: str, mode: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    path = Path(f"results/experiments/eval/{model_short_name}/results_{mode}.json")
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def generate_comparison_table(model_short_name: str, mode: str, dataset_list: list = None) -> str:
    """Generate markdown comparison table for baseline, blur, dyslexify, and combined."""
    
    if dataset_list is None:
        dataset_list = ["imagenet-100-typo", "imagenet-100-blur"]
    
    results = load_results(model_short_name, mode)
    
    if results is None:
        return "Error: Results not found. Please run evaluations first."
    
    imagenet_typo = results.get("ImageNet100", {})
    imagenet_blur = results.get("ImageNet100Blurred", {})
    
    baseline_acc = imagenet_typo.get("accuracy", 0) * 100
    baseline_typo_acc = imagenet_typo.get("typo_accuracy", 0) * 100
    
    blur_acc = imagenet_blur.get("accuracy", 0) * 100
    blur_typo_acc = imagenet_blur.get("typo_accuracy", 0) * 100
    
    dyslexify_acc = imagenet_typo.get("dislexified_accuracy", 0) * 100
    dyslexify_typo_acc = imagenet_typo.get("dislexified_typo_accuracy", 0) * 100
    
    blur_dyslexify_acc = imagenet_blur.get("dislexified_accuracy", 0) * 100
    blur_dyslexify_typo_acc = imagenet_blur.get("dislexified_typo_accuracy", 0) * 100
    
    table = f"""# Baseline Comparison: {model_short_name} ({mode} mode)

## ImageNet-100-Typo Results

| Method | Standard Acc | Typo Acc |
|--------|--------------|----------|
| Baseline | {baseline_acc:.2f}% | {baseline_typo_acc:.2f}% |
| Blur | {blur_acc:.2f}% | {blur_typo_acc:.2f}% |
| Dyslexify | {dyslexify_acc:.2f}% | {dyslexify_typo_acc:.2f}% |
| Blur+Dyslexify | {blur_dyslexify_acc:.2f}% | {blur_dyslexify_typo_acc:.2f}% |

## Analysis

- **Baseline**: Original model performance
- **Blur**: OCR-based text blurring applied to images
- **Dyslexify**: Attention head ablation defense
- **Blur+Dyslexify**: Combined approach using both methods

### Improvements over Baseline (Typo Acc)

- Blur: {blur_typo_acc - baseline_typo_acc:+.2f} percentage points
- Dyslexify: {dyslexify_typo_acc - baseline_typo_acc:+.2f} percentage points
- Blur+Dyslexify: {blur_dyslexify_typo_acc - baseline_typo_acc:+.2f} percentage points
"""
    
    return table


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline comparison table")
    parser.add_argument("--model", type=str, default="vit-b", help="Model short name")
    parser.add_argument("--mode", type=str, default="cls", help="Evaluation mode")
    args = parser.parse_args()
    
    table = generate_comparison_table(args.model, args.mode)
    
    output_path = Path(f"results/experiments/eval/baseline_comparison_{args.model}_{args.mode}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(table)
    
    print(table)
    print(f"\nTable saved to: {output_path}")


if __name__ == "__main__":
    main()

