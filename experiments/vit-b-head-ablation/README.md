# Head Ablation Experiment

This experiment systematically evaluates the impact of each attention head in a vision transformer model by ablating (disabling) each head individually and measuring the change in zero-shot classification accuracy for both normal and typographic images.

## Overview

The head ablation experiment helps identify which attention heads are most important for model performance by measuring the degradation in accuracy when each head is individually disabled. This provides insights into:

- Which heads are critical for normal image classification
- Which heads are important for handling typographic perturbations
- The relative importance of different attention heads across layers

## Key Features

- **Individual Head Evaluation**: Each attention head is ablated one at a time
- **Dual Accuracy Measurement**: Measures impact on both normal and typographic image accuracy
- **Efficient Sampling**: Uses a subset of 2,500 samples for faster evaluation
- **Comprehensive Results**: Provides detailed JSON output with statistics and rankings
- **Reproducible**: Uses fixed random seed for consistent subset sampling

## Usage

### Basic Usage

```bash
python experiments/vit-b-head-ablation/head_ablation.py
```

### Advanced Usage

```bash
python experiments/vit-b-head-ablation/head_ablation.py \
    --model vit-b \
    --device cuda:0 \
    --batch-size 512 \
    --num-workers 4 \
    --subset-size 2500 \
    --subset-seed 42 \
    --dataset-root /datasets/imagenet-100-typo \
    --save-dir results/experiments/vit-b-head-ablation \
    --verbose
```

### Command Line Arguments

- `--model`: Model short name (default: vit-b)
- `--device`: Device to run on (default: cuda:0)
- `--batch-size`: Batch size for data loading (default: 512)
- `--num-workers`: Number of data loading workers (default: 4)
- `--subset-size`: Number of samples to use (default: 2500)
- `--subset-seed`: Random seed for subset sampling (default: 42)
- `--dataset-root`: Dataset root directory (default: /datasets/imagenet-100-typo)
- `--save-dir`: Directory to save results (default: results/experiments/vit-b-head-ablation)
- `--verbose`: Enable verbose output

## Output

The experiment generates two JSON files:

### 1. `head_ablation_results.json`

Contains detailed results for each attention head:

```json
[
  {
    "layer": 0,
    "head": 5,
    "head_index": 5,
    "baseline_normal_acc": 0.8234,
    "baseline_typo_acc": 0.7123,
    "ablated_normal_acc": 0.8156,
    "ablated_typo_acc": 0.6987,
    "normal_delta": -0.0078,
    "typo_delta": -0.0136,
    "total_delta": -0.0214
  }
]
```

### 2. `head_ablation_summary.json`

Contains summary statistics and rankings:

```json
{
  "model": "vit-b",
  "num_layers": 12,
  "num_heads": 12,
  "total_heads": 144,
  "subset_size": 2500,
  "subset_seed": 42,
  "baseline_normal_acc": 0.8234,
  "baseline_typo_acc": 0.7123,
  "top_10_most_impactful_heads": [...],
  "top_10_least_impactful_heads": [...],
  "statistics": {
    "mean_normal_delta": -0.0023,
    "mean_typo_delta": -0.0041,
    "std_normal_delta": 0.0089,
    "std_typo_delta": 0.0123,
    "max_normal_delta": 0.0156,
    "min_normal_delta": -0.0234,
    "max_typo_delta": 0.0187,
    "min_typo_delta": -0.0345
  }
}
```

## Interpretation

### Delta Values

- **Positive delta**: Ablating the head improves accuracy
- **Negative delta**: Ablating the head decreases accuracy
- **Larger absolute values**: The head has more impact on performance

### Key Metrics

- **normal_delta**: Change in accuracy for normal images
- **typo_delta**: Change in accuracy for typographic images  
- **total_delta**: Combined impact (normal_delta + typo_delta)

### Rankings

- **Most impactful heads**: Heads whose ablation causes the largest accuracy drop
- **Least impactful heads**: Heads whose ablation has minimal effect on accuracy

## Performance Considerations

- **Subset size**: Using 2,500 samples provides a good balance between speed and accuracy
- **Batch size**: Adjust based on available GPU memory
- **Workers**: Increase for faster data loading if CPU cores are available
- **Device**: Use GPU for faster evaluation

## Dependencies

- `torch`
- `open_clip`
- `dyslexify` package
- ImageNet-100 dataset with typographic perturbations

## Example Analysis

After running the experiment, you can analyze the results to:

1. **Identify critical heads**: Find heads with large negative deltas
2. **Find redundant heads**: Identify heads with minimal impact
3. **Compare normal vs typo**: See which heads are more important for typographic robustness
4. **Layer analysis**: Examine if certain layers contain more important heads

This analysis can inform model pruning, interpretability studies, and robustness improvements.

