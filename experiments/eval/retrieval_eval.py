import open_clip
from dyslexify.config import MODELS
from typing import List, Dict, Tuple
from torch import nn
from typing import Any
from dyslexify.dataset import (
    ImageNet100,
    Food101,
    FGVCAircraft,
    ISIC2019Binary,
    SCAM,
    RTA100,
    Disentangling,
    Melanoma,
    ChestXRay,
    Paint,
)
from torch.utils.data import DataLoader
from dyslexify.zeroshot import calculate_text_features, zeroshot_retrieval
from dyslexify.defend import dislexify_openclip_model
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


class RetrievalEval:
    def __init__(
        self,
        model_short_name: str,
        device: str,
        dataset_list: List[str],
        mode: str,
        top_k: int = 10,
    ):
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.model_short_name = model_short_name
        self.device = device
        self.mode = mode
        self.top_k = top_k
        self.model, _, self.transform = self.load_model(device)
        self.dislexified_model, _, self.transform = self.load_model(device)
        typographic_attention_heads = self.load_typographic_attention_heads(
            model_short_name
        )
        self.dislexified_model = dislexify_openclip_model(
            self.dislexified_model,
            typographic_attention_heads=typographic_attention_heads,
            mode=self.mode,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.dataset_list = dataset_list
        self.datasets, self.dataloaders, self.raw_datasets = self.setup_dataset(
            self.dataset_list
        )

        torch.set_float32_matmul_precision("high")

    def load_model(self, device: str) -> Tuple[nn.Module, Any, Any]:
        model, tokenizer, transform = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )
        model.to(device)
        return model, tokenizer, transform

    def load_typographic_attention_heads(
        self, model_short_name: str
    ) -> List[Tuple[int, int]]:
        path = f"results/experiments/greedy_selection/{model_short_name}/ablated_heads_{self.mode}.json"
        with open(path, "r") as f:
            return json.load(f)

    def setup_dataset(self, dataset_list: List[str]):
        datasets = []
        dataloaders = []
        raw_datasets = []
        for dataset_name in dataset_list:
            if dataset_name == "imagenet-100-typo":
                dataset = ImageNet100(
                    root="/datasets/imagenet-100-typo",
                    split="val",
                    preprocess=self.transform,
                )
                raw_dataset = ImageNet100(
                    root="/datasets/imagenet-100-typo",
                    split="val",
                    preprocess=None,
                )
            elif dataset_name == "food101":
                dataset = Food101(
                    root="/datasets/food101",
                    split="test",
                    preprocess=self.transform,
                )
                raw_dataset = Food101(
                    root="/datasets/food101",
                    split="test",
                    preprocess=None,
                )
            elif dataset_name == "fgvc-aircraft":
                dataset = FGVCAircraft(
                    root="/datasets/fgvc-aircraft",
                    split="test",
                    download=True,
                    preprocess=self.transform,
                )
                raw_dataset = FGVCAircraft(
                    root="/datasets/fgvc-aircraft",
                    split="test",
                    download=True,
                    preprocess=None,
                )
            elif dataset_name == "isic2019":
                dataset = ISIC2019Binary(
                    root="/datasets/isic2019_typo",
                    preprocess=self.transform,
                )
                raw_dataset = ISIC2019Binary(
                    root="/datasets/isic2019_typo",
                    preprocess=None,
                )
            elif dataset_name == "scam":
                dataset = SCAM(
                    root="/datasets/scam",
                    preprocess=self.transform,
                )
                raw_dataset = SCAM(
                    root="/datasets/scam",
                    preprocess=None,
                )
            elif dataset_name == "rta100":
                dataset = RTA100(
                    root="/datasets/rta100",
                    preprocess=self.transform,
                )
                raw_dataset = RTA100(
                    root="/datasets/rta100",
                    preprocess=None,
                )
            elif dataset_name == "disentangling":
                dataset = Disentangling(
                    root="/datasets/disentangling",
                    preprocess=self.transform,
                )
                raw_dataset = Disentangling(
                    root="/datasets/disentangling",
                    preprocess=None,
                )
            elif dataset_name == "melanoma":
                dataset = Melanoma(
                    root="/datasets/melanoma_cancer_dataset_typo",
                    split="test",
                    preprocess=self.transform,
                )
                raw_dataset = Melanoma(
                    root="/datasets/melanoma_cancer_dataset_typo",
                    split="test",
                    preprocess=None,
                )
            elif dataset_name == "chest-xray":
                dataset = ChestXRay(
                    root="/datasets/chest_xray_typo",
                    split="test",
                    preprocess=self.transform,
                )
                raw_dataset = ChestXRay(
                    root="/datasets/chest_xray_typo",
                    split="test",
                    preprocess=None,
                )
            elif dataset_name == "paint":
                dataset = Paint(
                    root="/datasets/paint_ds",
                    preprocess=self.transform,
                )
                raw_dataset = Paint(
                    root="/datasets/paint_ds",
                    preprocess=None,
                )
            else:
                raise ValueError(f"Dataset {dataset_name} not found")

            dataloader = DataLoader(
                dataset, batch_size=128, shuffle=False, num_workers=16
            )
            datasets.append(dataset)
            dataloaders.append(dataloader)
            raw_datasets.append(raw_dataset)

        return datasets, dataloaders, raw_datasets

    def load_or_run_experiment(self):
        output_path = Path(
            f"results/experiments/eval/{self.model_short_name}/retrieval_results_{self.mode}_top{self.top_k}.json"
        )
        if output_path.exists():
            return json.load(open(output_path))
        return self.run_experiment()

    @torch.inference_mode()
    def run_experiment(self):
        all_results = {}

        for dataset, dataloader in zip(self.datasets, self.dataloaders):
            dataset_name = dataset.__class__.__name__
            print(f"Running retrieval evaluation on {dataset_name}")

            text_features = calculate_text_features(
                self.model, dataset, self.tokenizer, self.device
            )

            # Regular model retrieval
            regular_top_k, typo_top_k, mixed_top_k = zeroshot_retrieval(
                self.model, dataloader, text_features, self.device, self.top_k
            )

            # Dislexified model retrieval
            dislex_regular_top_k, dislex_typo_top_k, dislex_mixed_top_k = (
                zeroshot_retrieval(
                    self.dislexified_model,
                    dataloader,
                    text_features,
                    self.device,
                    self.top_k,
                )
            )

            # Convert results to serializable format (convert class indices to strings)
            dataset_results = {
                "regular_model": {
                    "regular_images": {str(k): v for k, v in regular_top_k.items()},
                    "typo_images": {str(k): v for k, v in typo_top_k.items()},
                    "mixed_images": {str(k): v for k, v in mixed_top_k.items()},
                },
                "dislexified_model": {
                    "regular_images": {
                        str(k): v for k, v in dislex_regular_top_k.items()
                    },
                    "typo_images": {str(k): v for k, v in dislex_typo_top_k.items()},
                    "mixed_images": {str(k): v for k, v in dislex_mixed_top_k.items()},
                },
                "num_classes": len(dataset.classes),
                "class_names": dataset.classes,
            }

            all_results[dataset_name] = dataset_results

            print(
                f"Completed retrieval for {dataset_name} - {len(dataset.classes)} classes"
            )

            del text_features

        # Save results
        output_path = Path(
            f"results/experiments/eval/{self.model_short_name}/retrieval_results_{self.mode}_top{self.top_k}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"Saved retrieval results to {output_path}")

        # Generate plots
        self.plot_top_images(all_results)

        return all_results

    def get_image_from_dataset(self, dataset, img_id: int, is_typo: bool = False):
        # Get images from dataset
        original_img, typo_img, _, _ = dataset[img_id]

        # Return the appropriate image
        if is_typo:
            return typo_img
        else:
            return original_img

    def plot_top_images(self, results: Dict):
        """Create comprehensive PDF with top 3 images for each class and configuration."""
        pdf_path = Path(
            f"results/experiments/eval/{self.model_short_name}/retrieval_plots_{self.mode}_top3.pdf"
        )
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(pdf_path) as pdf:
            for dataset_name, dataset_results in results.items():
                dataset = next(
                    d for d in self.datasets if d.__class__.__name__ == dataset_name
                )
                num_classes = dataset_results["num_classes"]
                class_names = dataset_results["class_names"]

                # Create one page per dataset
                self._plot_dataset_page(
                    pdf, dataset_name, dataset_results, dataset, class_names
                )

        print(f"Saved retrieval plots to {pdf_path}")

    def _plot_dataset_page(
        self,
        pdf,
        dataset_name: str,
        dataset_results: Dict,
        dataset,
        class_names: List[str],
    ):
        """Plot one page for a dataset showing only regular images configurations."""
        # Only show regular images for both models
        configs = [
            ("Regular Model - Regular Images", "regular_model", "regular_images"),
            (
                "Dislexified Model - Regular Images",
                "dislexified_model",
                "regular_images",
            ),
        ]

        # Select up to 5 evenly spaced classes
        total_classes = len(class_names)
        max_classes = min(5, total_classes)

        if total_classes <= 5:
            # Use all classes if we have 5 or fewer
            selected_indices = list(range(total_classes))
        else:
            # Select evenly spaced classes
            step = total_classes / max_classes
            selected_indices = [int(i * step) for i in range(max_classes)]

        selected_class_names = [class_names[i] for i in selected_indices]
        num_classes = len(selected_class_names)
        top_k = 3  # Show top 3 images

        # Create figure - optimized size for compact layout
        # Calculate based on actual content: 1 label column + 2 configs * 3 images = 7 columns
        total_cols = len(configs) * top_k + 1  # 7 columns total
        fig_width = total_cols * 1.5  # ~10.5 width for 7 columns
        fig_height = num_classes * 1.8  # Much more compact height per class
        fig, axes = plt.subplots(
            num_classes,
            total_cols,
            figsize=(fig_width, fig_height),
        )

        # Handle single class case
        if num_classes == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            f"Top 3 Retrieved Regular Images - {dataset_name}",
            fontsize=16,
            fontweight="bold",
        )

        for plot_idx, (class_idx, class_name) in enumerate(
            zip(selected_indices, selected_class_names)
        ):
            # Add class name in first column
            axes[plot_idx, 0].text(
                0.5,
                0.5,
                f"Class {class_idx}\n{class_name}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                wrap=True,
            )
            axes[plot_idx, 0].set_xlim(0, 1)
            axes[plot_idx, 0].set_ylim(0, 1)
            axes[plot_idx, 0].axis("off")

            col_offset = 1
            for config_idx, (config_name, model_type, image_type) in enumerate(configs):
                # Get top 3 images for this class and configuration
                class_data = dataset_results[model_type][image_type].get(
                    str(class_idx), []
                )
                top_3 = class_data[:top_k]

                # Plot top 3 images for this configuration
                for img_idx in range(top_k):
                    col = col_offset + config_idx * top_k + img_idx

                    if img_idx < len(top_3):
                        img_id, similarity = top_3[img_idx]

                        # For regular images, img_id is always positive (no typo images)
                        actual_img_id = img_id
                        is_typo = False  # Only regular images in this mode

                        # Load and display image from the correct raw dataset
                        raw_dataset = next(
                            d
                            for d in self.raw_datasets
                            if d.__class__.__name__ == dataset_name
                        )
                        img = self.get_image_from_dataset(
                            raw_dataset, actual_img_id, is_typo
                        )
                        axes[plot_idx, col].imshow(img)

                        # Get the real label of the image (always show real label, never typo)
                        try:
                            # Get the full sample data to extract labels
                            _, _, real_label_idx, typo_label_idx = raw_dataset[
                                actual_img_id
                            ]

                            # Always use the real label, never the typo label
                            if hasattr(raw_dataset, "classes"):
                                real_label_name = raw_dataset.classes[real_label_idx]
                            else:
                                real_label_name = f"Class_{real_label_idx}"

                        except (IndexError, AttributeError):
                            real_label_name = "Unknown"

                        # Add title with similarity score and real label only
                        title = f"{similarity:.3f}\n{real_label_name}"
                        axes[plot_idx, col].set_title(title, fontsize=8)

                    else:
                        # No image available
                        axes[plot_idx, col].text(
                            0.5, 0.5, "No\nimage", ha="center", va="center", fontsize=8
                        )

                    axes[plot_idx, col].axis("off")

                    # Add configuration header only for first class
                    if plot_idx == 0:
                        if img_idx == 1:  # Middle image of the 3
                            axes[plot_idx, col].text(
                                0.5,
                                1.15,  # Move header further up to avoid interference
                                config_name,
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                fontweight="bold",
                                transform=axes[plot_idx, col].transAxes,
                            )

        plt.tight_layout(pad=0.5)  # Reduce padding to minimize whitespace
        plt.subplots_adjust(
            top=0.92, hspace=0.05, wspace=0.1
        )  # Minimize vertical spacing between classes
        pdf.savefig(fig, bbox_inches="tight", dpi=800)  # Higher DPI for better quality
        plt.close(fig)


if __name__ == "__main__":
    eval = RetrievalEval(
        model_short_name="vit-b",
        device="cuda:1",
        dataset_list=["rta100"],
        mode="cls",
        top_k=10,
    )
    eval.run_experiment()
