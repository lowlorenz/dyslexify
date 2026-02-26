# %%
"""
Activation caching module for OpenCLIP models.

This module provides functionality to cache and extract activations from OpenCLIP
vision transformer models. It supports capturing residual activations at different
stages of the transformer blocks and attention patterns.

Classes:
    ActivationCache: Abstract base class defining the interface for activation caching.
    OpenClipActivationCache: Main class for caching activations from OpenCLIP models.
"""
from __future__ import annotations

from dyslexify.config import MODELS
import torch.nn as nn
import torch
from nnsight import NNsight
import open_clip
from abc import ABC, abstractmethod
from typing import List

from dyslexify.cache.cache import (
    ResidualCache,
    AttentionCache,
    LabeledResidualCache,
    LabeledAttentionCache,
    TypoLabeledResidualCache,
    # Legacy classes for backward compatibility
    ActivationCache,
    LabeledActivationCache,
    TypoLabeledActivationCache,
)
from dyslexify.cache.block import BlockResidual, BlockAttention
from dyslexify.cache.multi_head_attention import MultiheadAttentionWithWeightHook
from dyslexify.cache.hooks import create_layer_specific_attn_capture_hook


class ActivationCollector(ABC):
    """
    Abstract base class defining the interface for activation caching.

    This class defines the contract that all activation caching implementations
    must follow. It provides abstract methods for the core functionality of
    extracting and caching model activations.

    Subclasses should implement the specific logic for different model architectures
    and activation extraction strategies.
    """

    @abstractmethod
    def get_residual_cache(self, inputs: torch.Tensor) -> ResidualCache:
        """
        Get residual stream activations from the model for given inputs.

        Args:
            inputs: Input tensor to the model.

        Returns:
            ResidualCache containing all residual activations from the forward pass.
        """
        pass

    @abstractmethod
    def get_attention_cache(self, inputs: torch.Tensor) -> AttentionCache:
        """
        Get attention patterns from the model for given inputs.

        Args:
            inputs: Input tensor to the model.

        Returns:
            AttentionCache containing all attention patterns from the forward pass.
        """
        pass

    @abstractmethod
    def get_labeled_residual_cache(
        self, inputs: torch.Tensor, labels: List[str]
    ) -> LabeledResidualCache:
        """
        Get residual stream activations from the model for given inputs, with labels.
        """
        pass

    @abstractmethod
    def get_labeled_attention_cache(
        self, inputs: torch.Tensor, labels: List[str]
    ) -> LabeledAttentionCache:
        """
        Get attention patterns from the model for given inputs, with labels.
        """
        pass

    @abstractmethod
    def get_typo_labeled_residual_cache(
        self, inputs: torch.Tensor, labels: List[int], typo_labels: List[int]
    ) -> TypoLabeledResidualCache:
        """
        Get residual stream activations from the model for given inputs, with labels and typo labels.
        """
        pass

    # Legacy methods for backward compatibility
    @abstractmethod
    def get_activation_cache(self, inputs: torch.Tensor) -> ActivationCache:
        """
        Get activations from the model for given inputs.
        """
        pass

    @abstractmethod
    def get_labeled_activation_cache(
        self, inputs: torch.Tensor, labels: List[str]
    ) -> LabeledActivationCache:
        """
        Get activations from the model for given inputs, with labels.
        """
        pass

    @abstractmethod
    def get_typo_labeled_activation_cache(
        self, inputs: torch.Tensor, labels: List[int], typo_labels: List[int]
    ) -> TypoLabeledActivationCache:
        """
        Get activations from the model for given inputs, with labels and typo labels.
        """
        pass


def change_attn_implementation_to_hookable(module: nn.Module) -> None:
    """
    Swap the class of all MultiheadAttention modules in the given module to
    MultiheadAttentionWithWeightHook.

    This function modifies the class of all MultiheadAttention modules in the
    given module to MultiheadAttentionWithWeightHook. This is useful for
    capturing attention patterns with the MultiheadAttentionWithWeightHook.

    Args:
        module: The module to modify.
    """
    for m in module.modules():
        if isinstance(m, nn.MultiheadAttention) and not isinstance(
            m, MultiheadAttentionWithWeightHook
        ):
            m.__class__ = MultiheadAttentionWithWeightHook  # <-- weights stay put
            # Initialize the hook attributes since __init__ is not called
            # Use the current API (lists of hooks)
            m.attention_pattern_hooks = []
            m.attn_result_hooks = []


class OpenClipActivationCollector(ActivationCollector):
    """
    A class for caching activations from OpenCLIP vision transformer models.

    This class provides methods to extract and cache activations from different
    layers of an OpenCLIP model, including residual activations and attention
    patterns. It uses NNsight for tracing and attention hooks for capturing
    attention patterns in a single forward pass.

    Attributes:
        model: The OpenCLIP model instance.
        nn_model: The NNsight-wrapped model for activation tracing.
        attention_hooks: Dictionary to store hooks for accessing captured patterns.
    """

    def __init__(self, model_name: str, pretrained: str, device: str) -> None:
        """
        Initialize the activation cache with an OpenCLIP model.

        Args:
            model_name: The name of the OpenCLIP model architecture.
            pretrained: The name of the pretrained weights to load.
        """
        self.model = open_clip.create_model(model_name, pretrained=pretrained)
        change_attn_implementation_to_hookable(self.model)
        self.model.eval()
        self.model.to(device)
        self.nn_model = NNsight(self.model.visual)
        self.attention_hooks = {}
        self.device = device

    def _setup_attention_capture_hooks(self, cls_token_only: bool = False):
        """
        Set up attention pattern capture hooks for all attention layers.

        Args:
            cls_token_only: Whether to capture only CLS token attention patterns.
        """

        # Clear any existing hooks
        self.attention_hooks.clear()

        # Find all attention layers and set up hooks
        for i, block in enumerate(self.model.visual.transformer.resblocks):
            if hasattr(block.attn, "register_attention_pattern_hook"):
                hook = create_layer_specific_attn_capture_hook(i, cls_token_only)
                block.attn.register_attention_pattern_hook(hook)
                self.attention_hooks[i] = hook

    def _cleanup_attention_hooks(self):
        """Remove all attention pattern capture hooks."""
        for block in self.model.visual.transformer.resblocks:
            if hasattr(block.attn, "remove_attention_pattern_hook"):
                block.attn.remove_attention_pattern_hook()

    @torch.no_grad()
    def get_residual_cache(
        self, inputs: torch.Tensor, cls_token_only=False
    ) -> ResidualCache:
        """
        Get residual stream activations from the model for given inputs.

        Uses NNsight to capture residual activations.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).
            cls_token_only: Whether to capture only CLS token activations.

        Returns:
            ResidualCache containing all residual activations from the forward pass.
        """
        inputs = inputs.to(self.device)

        token_slice = slice(0, 1) if cls_token_only else slice(None)

        with self.nn_model.trace(inputs, scan=False, validate={}):
            resblocks = self.nn_model.transformer.resblocks

            # Create ResidualCache instance
            residual_cache = ResidualCache()

            # Process each block
            for i, block in enumerate(resblocks):
                # Get residual activations
                residual_pre = block.input[:, token_slice].save()
                residual_mid = block.ln_2.input[:, token_slice].save()
                residual_post = block.output[:, token_slice].save()

                # Create block residual activations
                block_residual = BlockResidual(
                    residual_pre=residual_pre,
                    residual_mid=residual_mid,
                    residual_post=residual_post,
                )

                residual_cache.blocks.append(block_residual)

        return residual_cache

    @torch.no_grad()
    def get_attention_cache(
        self, inputs: torch.Tensor, cls_token_only=False
    ) -> AttentionCache:
        """
        Get attention patterns from the model for given inputs.

        Uses attention hooks to capture attention patterns.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).
            cls_token_only: Whether to capture only CLS token attention patterns.

        Returns:
            AttentionCache containing all attention patterns from the forward pass.
        """
        inputs = inputs.to(self.device)

        # Set up attention hooks
        self._setup_attention_capture_hooks(cls_token_only)

        try:
            # Run a forward pass to capture attention patterns via hooks
            with torch.no_grad():
                _ = self.model.visual(inputs)

            # Create AttentionCache instance
            attention_cache = AttentionCache()

            # Collect attention patterns
            for i in range(len(self.model.visual.transformer.resblocks)):
                if i not in self.attention_hooks or not hasattr(
                    self.attention_hooks[i], "captured_patterns"
                ):
                    raise ValueError(
                        f"Attention hook for block {i} not found in attention hooks"
                    )

                if i not in self.attention_hooks[i].captured_patterns:
                    raise ValueError(
                        f"Attention pattern for block {i} not found in attention hooks"
                    )

                attn_pattern = self.attention_hooks[i].captured_patterns[i]

                # Create block attention activations
                block_attention = BlockAttention(attn_pattern=attn_pattern)
                attention_cache.blocks.append(block_attention)

            return attention_cache

        finally:
            # Clean up attention hooks
            self._cleanup_attention_hooks()

    @torch.no_grad()
    def get_labeled_residual_cache(
        self, inputs: torch.Tensor, labels: List[int], cls_token_only=False
    ) -> LabeledResidualCache:
        """
        Get residual stream activations from the model for given inputs with labels.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).
            labels: List of labels for each input.
            cls_token_only: Whether to capture only CLS token activations.

        Returns:
            LabeledResidualCache containing all residual activations with labels.
        """
        residual_cache = self.get_residual_cache(inputs, cls_token_only)
        labeled_cache = LabeledResidualCache(
            blocks=residual_cache.blocks, labels=torch.tensor(labels)
        )
        return labeled_cache

    @torch.no_grad()
    def get_labeled_attention_cache(
        self, inputs: torch.Tensor, labels: List[int], cls_token_only=False
    ) -> LabeledAttentionCache:
        """
        Get attention patterns from the model for given inputs with labels.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).
            labels: List of labels for each input.
            cls_token_only: Whether to capture only CLS token attention patterns.

        Returns:
            LabeledAttentionCache containing all attention patterns with labels.
        """
        attention_cache = self.get_attention_cache(inputs, cls_token_only)
        labeled_cache = LabeledAttentionCache(
            blocks=attention_cache.blocks, labels=torch.tensor(labels)
        )
        return labeled_cache

    @torch.no_grad()
    def get_typo_labeled_residual_cache(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        typo_labels: List[int],
        cls_token_only=False,
    ) -> TypoLabeledResidualCache:
        """
        Get residual stream activations from the model for given inputs, with labels and typo labels.
        """
        residual_cache = self.get_residual_cache(inputs, cls_token_only)
        return TypoLabeledResidualCache.from_residual_cache(
            residual_cache, torch.tensor(labels), torch.tensor(typo_labels)
        )

    # Legacy methods for backward compatibility
    @torch.no_grad()
    def get_activation_cache(
        self, inputs: torch.Tensor, cls_token_only=False
    ) -> ActivationCache:
        """
        Get activations from the model for given inputs without attention patterns.

        Uses NNsight to capture residual activations.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            ActivationCache containing all block activations from the forward pass.
        """
        residual_cache = self.get_residual_cache(inputs, cls_token_only)
        return ActivationCache(blocks=residual_cache.blocks)

    @torch.no_grad()
    def get_labeled_activation_cache(
        self, inputs: torch.Tensor, labels: List[int], cls_token_only=False
    ) -> LabeledActivationCache:
        """
        Get activations from the model for given inputs with labels.

        Args:
            inputs: Input tensor of shape (batch_size, channels, height, width).
            labels: List of labels for each input.

        Returns:
            LabeledActivationCache containing all block activations with labels.
        """
        labeled_residual_cache = self.get_labeled_residual_cache(
            inputs, labels, cls_token_only
        )
        return LabeledActivationCache(
            blocks=labeled_residual_cache.blocks, labels=labeled_residual_cache.labels
        )

    @torch.no_grad()
    def get_typo_labeled_activation_cache(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        typo_labels: List[int],
        cls_token_only=False,
    ) -> TypoLabeledActivationCache:
        """
        Get activations from the model for given inputs, with labels and typo labels.
        """
        typo_labeled_residual_cache = self.get_typo_labeled_residual_cache(
            inputs, labels, typo_labels, cls_token_only
        )
        return TypoLabeledActivationCache.from_activation_cache(
            ActivationCache(blocks=typo_labeled_residual_cache.blocks),
            typo_labeled_residual_cache.labels,
            typo_labeled_residual_cache.typo_labels,
        )


if __name__ == "__main__":
    # Example usage
    collector = OpenClipActivationCollector(**MODELS["vit-b"])
    residual_cache = collector.get_residual_cache(torch.randn(1, 3, 224, 224))
    attention_cache = collector.get_attention_cache(torch.randn(1, 3, 224, 224))

    # Access activations using the new separate cache interfaces
    print(f"Number of residual blocks: {len(residual_cache)}")
    print(f"Number of attention blocks: {len(attention_cache)}")
    print(
        f"First residual block residual_pre shape: {residual_cache[0].residual_pre.shape}"
    )
    print(
        f"First attention block attn_pattern shape: {attention_cache[0].attn_pattern.shape}"
    )

    # Demonstrate the new __repr__ functionality
    print(f"\nResidualCache representation:")
    print(repr(residual_cache))
    print(f"\nAttentionCache representation:")
    print(repr(attention_cache))
    print(f"\nFirst residual block representation:")
    print(repr(residual_cache[0]))
    print(f"\nFirst attention block representation:")
    print(repr(attention_cache[0]))

# %%
