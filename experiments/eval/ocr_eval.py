import open_clip
from dyslexify.config import MODELS
from dyslexify.dataset.iiit5k import IIIT5K
from dyslexify.defend import dislexify_openclip_model
from torch.utils.data import DataLoader
import torch
import json
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def collate_fn(batch):
    """Custom collate function to handle variable-length lexicons"""
    images, ground_truths, small_lexis, medium_lexis = zip(*batch)
    images = torch.stack(images)
    return images, ground_truths, small_lexis, medium_lexis


class OCREval:
    def __init__(self, model_short_name: str, device: str, mode: str):
        self.model_name = MODELS[model_short_name]["model_name"]
        self.pretrained = MODELS[model_short_name]["pretrained"]
        self.model_short_name = model_short_name
        self.device = device
        self.mode = mode

        self.model, _, self.transform = self.load_model(device)
        self.dislexified_model, _, _ = self.load_model(device)

        typographic_attention_heads = self.load_typographic_attention_heads(
            model_short_name
        )
        self.dislexified_model = dislexify_openclip_model(
            self.dislexified_model,
            typographic_attention_heads=typographic_attention_heads,
            mode=self.mode,
        )

        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.dataset = IIIT5K(root="iiit5k", preprocess=self.transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
        )

        torch.set_float32_matmul_precision("high")

    def load_model(self, device: str):
        model, tokenizer, transform = open_clip.create_model_and_transforms(
            self.model_name, self.pretrained
        )
        model.to(device)
        return model, tokenizer, transform

    def load_typographic_attention_heads(
        self, model_short_name: str
    ) -> List[Tuple[int, int]]:
        path = f"iclr_results/experiments/greedy_selection/{model_short_name}/ablated_heads_{self.mode}.json"
        with open(path, "r") as f:
            return json.load(f)

    @torch.inference_mode()
    def evaluate_lexicon(
        self, model, image, ground_truth: str, lexicon: List[str], debug: bool = False
    ) -> bool:
        image = image.to(self.device)
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_tokens = self.tokenizer(lexicon).to(self.device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).squeeze(0)
        predicted_idx = similarity.argmax().item()
        predicted_word = lexicon[predicted_idx]

        if debug:
            print(f"Ground truth: '{ground_truth}'")
            print(f"Predicted: '{predicted_word}'")
            print(f"In lexicon: {ground_truth in lexicon}")
            print(
                f"Top 3 predictions: {[lexicon[i] for i in similarity.argsort(descending=True)[:3].tolist()]}"
            )

        return predicted_word == ground_truth

    @torch.inference_mode()
    def run_experiment(self, debug_first_n: int = 0):
        small_correct = 0
        medium_correct = 0
        dislex_small_correct = 0
        dislex_medium_correct = 0
        total = len(self.dataset)

        for idx, (image, ground_truth, small_lexi, medium_lexi) in enumerate(
            tqdm(self.dataloader)
        ):
            gt = ground_truth[0]
            small_lex = small_lexi[0]
            medium_lex = medium_lexi[0]

            debug = idx < debug_first_n

            if debug:
                print(f"\n=== Sample {idx} ===")
                print(f"Ground truth type: {type(gt)}, value: '{gt}'")
                print(f"Small lexicon size: {len(small_lex)}")
                print(f"GT in small lexicon: {gt in small_lex}")
                print(f"First 5 words in small lexicon: {small_lex[:5]}")

            if self.evaluate_lexicon(self.model, image, gt, small_lex, debug=debug):
                small_correct += 1

            if self.evaluate_lexicon(self.model, image, gt, medium_lex, debug=debug):
                medium_correct += 1

            if self.evaluate_lexicon(
                self.dislexified_model, image, gt, small_lex, debug=debug
            ):
                dislex_small_correct += 1

            if self.evaluate_lexicon(
                self.dislexified_model, image, gt, medium_lex, debug=debug
            ):
                dislex_medium_correct += 1

        results = {
            "small_lexicon": {
                "accuracy": small_correct / total,
                "dislexified_accuracy": dislex_small_correct / total,
            },
            "medium_lexicon": {
                "accuracy": medium_correct / total,
                "dislexified_accuracy": dislex_medium_correct / total,
            },
        }

        print(
            f"Small Lexicon - Accuracy: {results['small_lexicon']['accuracy']:.4f}, "
            f"Dislexified: {results['small_lexicon']['dislexified_accuracy']:.4f}"
        )
        print(
            f"Medium Lexicon - Accuracy: {results['medium_lexicon']['accuracy']:.4f}, "
            f"Dislexified: {results['medium_lexicon']['dislexified_accuracy']:.4f}"
        )

        output_path = Path(
            f"iclr_results/experiments/ocr_eval/{self.model_short_name}/results_{self.mode}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return results


if __name__ == "__main__":
    for model_short_name in MODELS.keys():
        if model_short_name == "vit-l":
            continue
        eval = OCREval(
            model_short_name=model_short_name,
            device="cuda:0",
            mode="cls",
        )
        eval.run_experiment(debug_first_n=0)
        del eval
