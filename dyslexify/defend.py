from torch import nn
from dyslexify.cache.hooks import (
    create_zero_cls_attention_result_hook,
    create_zero_spatial_attention_result_hook,
)
from dyslexify.cache.collector import change_attn_implementation_to_hookable
from typing import List, Tuple


def dislexify_openclip_model(
    model: nn.Module, typographic_attention_heads: List[Tuple[int, int]], mode: str
) -> nn.Module:
    """
    Dislexify an OpenCLIP model.

    Args:
        model (nn.Module): The OpenCLIP model to dyslexify.

    Returns:
        nn.Module: The dislexified OpenCLIP model.
    """

    change_attn_implementation_to_hookable(model)

    # register the suppression hooks
    blocks = model.visual.transformer.resblocks
    for layer_idx, head_idx in typographic_attention_heads:
        if mode == "cls":
            blocks[layer_idx].attn.register_attn_result_hook(
                create_zero_cls_attention_result_hook(head_idx)
            )
        elif mode == "spatial":
            blocks[layer_idx].attn.register_attn_result_hook(
                create_zero_spatial_attention_result_hook(head_idx)
            )

    return model
