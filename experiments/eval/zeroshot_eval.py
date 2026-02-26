import open_clip
from dyslexify.config import MODELS
from typing import List
from torch import nn
from typing import Tuple, Any
from dyslexify.dataset import (
    ImageNet100,
    ImageNet100Blurred,
    Food101,
    FGVCAircraft,
    ISIC2019Binary,
    SCAM,
    RTA100,
    Disentangling,
    Melanoma,
    ChestXRay,
    Paint,
    BCN20k,
    HAM10k,
)
from torch.utils.data import DataLoader
from dyslexify.zeroshot import calculate_text_features, zeroshot_classifier
from dyslexify.defend import dislexify_openclip_model
import torch
import json
from pathlib import Path


class ZeroShotEval:
    def __init__(
        self, model_short_name: str, device: str, dataset_list: List[str], mode: str
    ):
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.model_short_name = model_short_name
        self.device = device
        self.mode = mode
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
        self.datasets, self.dataloaders = self.setup_dataset(self.dataset_list)

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
        for dataset_name in dataset_list:
            if dataset_name == "imagenet-100-typo":
                dataset = ImageNet100(
                    root="/datasets/imagenet-100-typo",
                    split="val",
                    preprocess=self.transform,
                )
            elif dataset_name == "imagenet-100-blur":
                dataset = ImageNet100Blurred(
                    root="/datasets/imagenet-100-typo-blur",
                    split="val",
                    preprocess=self.transform,
                )
            elif dataset_name == "food101":
                dataset = Food101(
                    root="/datasets/food101",
                    split="test",
                    preprocess=self.transform,
                )
            elif dataset_name == "fgvc-aircraft":
                dataset = FGVCAircraft(
                    root="/datasets/fgvc-aircraft",
                    split="test",
                    download=True,
                    preprocess=self.transform,
                )
            elif dataset_name == "isic2019":
                dataset = ISIC2019Binary(
                    root="/datasets/isic2019_typo",
                    preprocess=self.transform,
                )
            elif dataset_name == "scam":
                dataset = SCAM(
                    root="/datasets/scam",
                    preprocess=self.transform,
                )
            elif dataset_name == "rta100":
                dataset = RTA100(
                    root="/datasets/rta100",
                    preprocess=self.transform,
                )
            elif dataset_name == "disentangling":
                dataset = Disentangling(
                    root="/datasets/disentangling",
                    preprocess=self.transform,
                )
            elif dataset_name == "melanoma":
                dataset = Melanoma(
                    root="/datasets/melanoma_cancer_dataset_typo",
                    split="test",
                    preprocess=self.transform,
                )
            elif dataset_name == "chest-xray":
                dataset = ChestXRay(
                    root="/datasets/chest_xray_typo",
                    split="test",
                    preprocess=self.transform,
                )
            elif dataset_name == "paint":
                dataset = Paint(
                    root="/datasets/paint_ds",
                    preprocess=self.transform,
                )
            elif dataset_name == "bcn":
                dataset = BCN20k(
                    root="/datasets/BCN_20k/",
                    preprocess=self.transform,
                )
            elif dataset_name == "ham10k":
                dataset = HAM10k(
                    root="/datasets/HAM",
                    preprocess=self.transform,
                )
            else:
                raise ValueError(f"Dataset {dataset_name} not found")

            dataloader = DataLoader(
                dataset, batch_size=128, shuffle=False, num_workers=16
            )
            datasets.append(dataset)
            dataloaders.append(dataloader)

        return datasets, dataloaders

    def load_or_run_experiment(self):
        output_path = Path(
            f"results/experiments/eval/{self.model_short_name}/results_{self.mode}.json"
        )
        if output_path.exists():
            return json.load(open(output_path))
        return self.run_experiment()

    @torch.inference_mode()
    def run_experiment(self):
        accs = []
        typo_accs = []
        dislex_accs = []
        dislex_typo_accs = []
        for dataset, dataloader in zip(self.datasets, self.dataloaders):
            text_features = calculate_text_features(
                self.model, dataset, self.tokenizer, self.device
            )
            acc, typo_acc = zeroshot_classifier(
                self.model, dataloader, text_features, self.device
            )
            dislex_acc, dislex_typo_acc = zeroshot_classifier(
                self.dislexified_model, dataloader, text_features, self.device
            )
            accs.append(acc)
            typo_accs.append(typo_acc)
            dislex_accs.append(dislex_acc)
            dislex_typo_accs.append(dislex_typo_acc)

            del text_features

        results = {}
        for dataset, acc, typo_acc, dislex_acc, dislex_typo_acc in zip(
            self.datasets, accs, typo_accs, dislex_accs, dislex_typo_accs
        ):
            print(
                f"Dataset: {dataset.__class__.__name__}, Accuracy: {acc:.4f}, Typo Accuracy: {typo_acc:.4f}, Dislexified Accuracy: {dislex_acc:.4f}, Dislexified Typo Accuracy: {dislex_typo_acc:.4f}"
            )
            results[dataset.__class__.__name__] = {
                "accuracy": acc,
                "typo_accuracy": typo_acc,
                "dislexified_accuracy": dislex_acc,
                "dislexified_typo_accuracy": dislex_typo_acc,
            }

        output_path = Path(
            f"results/experiments/eval/{self.model_short_name}/results_{self.mode}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    eval = ZeroShotEval(
        model_short_name="vit-b",
        device="cuda:1",
        dataset_list=["disentangling"],
        mode="cls",
    )
    # "imagenet-100-typo",
    # "food101",
    # "rta100"
    # "food101", "fgvc-aircraft", "isic2019"],
    # eval = NonTypoZeroShotEval(
    #     model_short_name="whylesionclip",
    #     device="cuda:7",
    #     dataset_list=["isic2019"],
    # )
    eval.run_experiment()
