MODELS = {
    "vit-b": {
        "model_name": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
    },
    "vit-l": {
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
    },
    "vit-h": {
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
    },
    "vit-g": {
        "model_name": "ViT-g-14",
        "pretrained": "laion2b_s34b_b88k",
    },
    "vit-big-g": {
        "model_name": "ViT-bigG-14",
        "pretrained": "laion2b_s39b_b160k",
    },
    "whylesionclip": {
        "model_name": "hf-hub:yyupenn/whylesionclip",
        "pretrained": "ViT-L-14",
    },
    "whyxrayclip": {
        "model_name": "hf-hub:yyupenn/whyxrayclip",
        "pretrained": "ViT-L-14",
    },
}

import torch

DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
