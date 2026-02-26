#!/usr/bin/env python
"""Run blur baseline evaluation for all OpenCLIP models."""
import subprocess
import sys
from pathlib import Path

# All OpenCLIP models to evaluate
MODELS = ["vit-b", "vit-l", "vit-h", "vit-g", "vit-big-g"]
MODE = "cls"
DEVICE = "cuda:0"  # You can change this if needed


def run_evaluation(model_short_name: str):
    """Run evaluation for a single model."""
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: {model_short_name}")
    print("=" * 80 + "\n")

    # Import here to avoid loading all models at once
    from zeroshot_eval import ZeroShotEval

    eval = ZeroShotEval(
        model_short_name=model_short_name,
        device=DEVICE,
        dataset_list=["imagenet-100-typo", "imagenet-100-blur"],
        mode=MODE,
    )

    try:
        results = eval.run_experiment()
        print(f"\n✓ {model_short_name} evaluation complete!")
        return True
    except Exception as e:
        print(f"\n✗ Error evaluating {model_short_name}: {e}")
        return False


def generate_all_comparison_tables():
    """Generate comparison tables for all models."""
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON TABLES")
    print("=" * 80 + "\n")

    for model in MODELS:
        results_path = Path(f"results/experiments/eval/{model}/results_{MODE}.json")
        if results_path.exists():
            print(f"\nGenerating table for {model}...")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "experiments/analysis/compare_baselines.py",
                    "--model",
                    model,
                    "--mode",
                    MODE,
                ]
            )
        else:
            print(f"\n⚠ No results found for {model}, skipping table generation")


def main():
    print("=" * 80)
    print("RUNNING BLUR BASELINE EVALUATION FOR ALL OPENCLIP MODELS")
    print("=" * 80)
    print(f"\nModels to evaluate: {', '.join(MODELS)}")
    print(f"Mode: {MODE}")
    print(f"Device: {DEVICE}")
    print("\nThis will take a while...")

    successful = []
    failed = []

    for model in MODELS:
        success = run_evaluation(model)
        if success:
            successful.append(model)
        else:
            failed.append(model)

    # Generate comparison tables
    generate_all_comparison_tables()

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n✓ Successful: {len(successful)}/{len(MODELS)}")
    if successful:
        print(f"  {', '.join(successful)}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(MODELS)}")
        print(f"  {', '.join(failed)}")

    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE!")
    print("=" * 80)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
