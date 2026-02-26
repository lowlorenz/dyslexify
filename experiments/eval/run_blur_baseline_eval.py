from zeroshot_eval import ZeroShotEval

model_short_name = "vit-b"
device = "cuda:1"
mode = "cls"

print("Running evaluation on ImageNet-100-typo and ImageNet-100-blur datasets...")
print("This will evaluate: Baseline, Blur, Dyslexify, and Blur+Dyslexify")

eval = ZeroShotEval(
    model_short_name=model_short_name,
    device=device,
    dataset_list=["imagenet-100-typo", "imagenet-100-blur"],
    mode=mode,
)

results = eval.run_experiment()

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
print(
    "\nResults saved to:",
    f"results/experiments/eval/{model_short_name}/results_{mode}.json",
)

