"""
This module contains the ActivationCache class, which is a container for all block activations in one forward pass.
"""

from __future__ import annotations

import torch

# from nnsight.intervention.graph import InterventionProxy
from typing import Optional, List, TypeVar, Iterator
from dataclasses import dataclass, field
from abc import ABC
from dyslexify.cache.block import BlockResidual, BlockAttention, T

A = TypeVar("A", bound="ActivationCacheBase")


@dataclass
class ActivationCacheBase(ABC):
    """Abstract base class for all activation caches."""

    blocks: List[T] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self) -> Iterator[T]:
        return iter(self.blocks)

    def __getitem__(self, idx: int) -> T:
        return self.blocks[idx]

    def __repr__(self) -> str:
        """Return a string representation showing field names and tensor shapes."""
        if not self.blocks:
            return f"{type(self).__name__}(blocks=[])"

        # Get the first block to show field structure
        first_block = self.blocks[0]
        field_info = first_block._get_repr_info()

        return f"{type(self).__name__}(blocks={len(self.blocks)}, fields=[{', '.join(field_info)}])"

    def to(self, device: torch.device | str, **kwargs):
        for blk in self.blocks:
            blk.to(device, **kwargs)
        return self

    def concatenate(self, other: A) -> A:
        """Concatenate two activation caches along the batch dimension."""
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot concatenate {type(other)} with {type(self).__name__}"
            )

        if len(self.blocks) != len(other.blocks):
            raise ValueError(
                f"Cannot concatenate {type(self).__name__} with {type(other).__name__} because they have different lengths"
            )

        for i in range(len(self.blocks)):
            self.blocks[i].concatenate(other.blocks[i])
        return self

    def save(self, path: str, cpu: bool = True) -> None:
        """torch.save the cache; move to CPU first if wanted."""
        obj = self.to("cpu") if cpu else self
        torch.save(obj, path)

    @classmethod
    def load(cls, path: str, map_location="cpu") -> A:
        torch.serialization.add_safe_globals(
            [
                ActivationCache,
                ResidualCache,
                AttentionCache,
                BlockResidual,
                BlockAttention,
            ]
        )
        return torch.load(path, map_location=map_location, weights_only=True)


@dataclass
class ResidualCache(ActivationCacheBase):
    """Container for all residual stream activations in one forward pass."""

    blocks: List[BlockResidual] = field(default_factory=list)


@dataclass
class AttentionCache(ActivationCacheBase):
    """Container for all attention patterns in one forward pass."""

    blocks: List[BlockAttention] = field(default_factory=list)


# Legacy class for backward compatibility
@dataclass
class ActivationCache(ResidualCache):
    """Legacy container for all block activations in one forward pass."""

    blocks: List[BlockResidual] = field(default_factory=list)


@dataclass(kw_only=True)
class LabeledResidualCache(ResidualCache):
    """Container for all residual stream activations in one forward pass, with labels."""

    labels: torch.Tensor

    def concatenate(self, other: LabeledResidualCache) -> LabeledResidualCache:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot concatenate {type(other)} with {type(self).__name__}"
            )
        # Call parent class concatenate to properly concatenate tensors within blocks
        super().concatenate(other)
        self.labels = torch.cat([self.labels, other.labels], dim=0)
        return self

    @classmethod
    def from_residual_cache(
        cls,
        residual_cache: ResidualCache,
        labels: torch.Tensor,
    ) -> LabeledResidualCache:
        return cls(
            blocks=residual_cache.blocks,
            labels=labels,
        )


@dataclass(kw_only=True)
class LabeledAttentionCache(AttentionCache):
    """Container for all attention patterns in one forward pass, with labels."""

    labels: torch.Tensor

    def concatenate(self, other: LabeledAttentionCache) -> LabeledAttentionCache:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot concatenate {type(other)} with {type(self).__name__}"
            )
        # Call parent class concatenate to properly concatenate tensors within blocks
        super().concatenate(other)
        self.labels = torch.cat([self.labels, other.labels], dim=0)
        return self

    @classmethod
    def from_attention_cache(
        cls,
        attention_cache: AttentionCache,
        labels: torch.Tensor,
    ) -> LabeledAttentionCache:
        return cls(
            blocks=attention_cache.blocks,
            labels=labels,
        )


# Legacy labeled classes for backward compatibility
@dataclass(kw_only=True)
class LabeledActivationCache(LabeledResidualCache):
    """Legacy container for all block activations in one forward pass, with labels."""

    blocks: List[BlockResidual] = field(default_factory=list)
    labels: torch.Tensor

    @classmethod
    def from_activation_cache(
        cls,
        activation_cache: ActivationCache,
        labels: torch.Tensor,
    ) -> LabeledActivationCache:
        return cls(
            blocks=activation_cache.blocks,
            labels=labels,
        )


@dataclass(kw_only=True)
class TypoLabeledResidualCache(ResidualCache):
    """Container for all residual stream activations in one forward pass, with labels and typo labels."""

    labels: torch.Tensor
    typo_labels: torch.Tensor

    def concatenate(self, other: TypoLabeledResidualCache) -> TypoLabeledResidualCache:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot concatenate {type(other)} with {type(self).__name__}"
            )
        # Call parent class concatenate to properly concatenate tensors within blocks
        super().concatenate(other)
        self.labels = torch.cat([self.labels, other.labels], dim=0)
        self.typo_labels = torch.cat([self.typo_labels, other.typo_labels], dim=0)
        return self

    @classmethod
    def from_residual_cache(
        cls,
        residual_cache: ResidualCache,
        labels: torch.Tensor,
        typo_labels: torch.Tensor,
    ) -> TypoLabeledResidualCache:
        return cls(
            blocks=residual_cache.blocks,
            labels=labels,
            typo_labels=typo_labels,
        )


# Legacy typo labeled class for backward compatibility
@dataclass(kw_only=True)
class TypoLabeledActivationCache(TypoLabeledResidualCache):
    """Legacy container for all block activations in one forward pass, with labels and typo labels."""

    blocks: List[BlockResidual] = field(default_factory=list)
    labels: torch.Tensor
    typo_labels: torch.Tensor

    @classmethod
    def from_activation_cache(
        cls,
        activation_cache: ActivationCache,
        labels: torch.Tensor,
        typo_labels: torch.Tensor,
    ) -> TypoLabeledActivationCache:
        return cls(
            blocks=activation_cache.blocks,
            labels=labels,
            typo_labels=typo_labels,
        )

    @classmethod
    def from_labeled_activation_cache(
        cls, labeled_cache: LabeledActivationCache, typo_labels: torch.Tensor
    ) -> TypoLabeledActivationCache:
        return cls(
            blocks=labeled_cache.blocks,
            labels=labeled_cache.labels,
            typo_labels=typo_labels,
        )
