"""
This module contains the ActivationCache class, which is a container for all block activations in one forward pass.
"""

from __future__ import annotations

import torch

# from nnsight.intervention.graph import InterventionProxy
from typing import List, TypeVar
from dataclasses import dataclass, fields, field
from abc import ABC


def empty_tensor() -> torch.Tensor:
    """Default factory that creates an empty torch tensor."""
    return torch.empty(0)


T = TypeVar("T", bound="BlockActivationsBase")


class BlockActivationsBase(ABC):
    """Abstract base class for all block activations."""

    def __getitem__(self, name: str) -> torch.Tensor | None:
        """Get activation by name."""
        return getattr(self, name)

    def to(self, device: torch.device | str, **kwargs):
        """Move tensors to device."""
        for f in fields(self):
            tensor = getattr(self, f.name)
            if isinstance(tensor, torch.Tensor) or isinstance(
                tensor, InterventionProxy
            ):
                setattr(self, f.name, tensor.to(device, **kwargs))
        return self

    def _concatenate_tensors(self, other: BlockActivationsBase) -> None:
        """Helper method to concatenate tensors along batch dimension."""
        for f in fields(self):
            tensor = getattr(self, f.name)
            other_tensor = getattr(other, f.name)
            if tensor is not None and other_tensor is not None:
                # Check that all dims except batch (dim 0) match
                assert (
                    tensor.shape[1:] == other_tensor.shape[1:]
                ), f"Shape mismatch in field '{f.name}': {tensor.shape} vs {other_tensor.shape}"
                setattr(self, f.name, torch.cat([tensor, other_tensor], dim=0))
            else:
                # If not a tensor or one is None, just keep the original
                setattr(self, f.name, tensor)

    def _get_repr_info(self) -> List[str]:
        """Helper method to get field information for __repr__."""
        field_info = []
        for field_name in fields(self):
            tensor = getattr(self, field_name.name)
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    field_info.append(f"{field_name.name}: {list(tensor.shape)}")
                elif hasattr(
                    tensor, "shape"
                ):  # Handle InterventionProxy and similar objects
                    field_info.append(f"{field_name.name}: {list(tensor.shape)}")
                else:
                    field_info.append(f"{field_name.name}: {type(tensor).__name__}")
            else:
                field_info.append(f"{field_name.name}: None")
        return field_info

    def concatenate(self, other: T) -> T:
        """Concatenate two block activations along the batch dimension."""
        if not isinstance(other, type(self)):
            raise TypeError(
                f"Cannot concatenate {type(other)} with {type(self).__name__}"
            )

        self._concatenate_tensors(other)
        return self

    def __repr__(self) -> str:
        """Return a string representation showing field names and tensor shapes."""
        field_info = self._get_repr_info()
        return f"{type(self).__name__}({', '.join(field_info)})"


@dataclass
class BlockResidual(BlockActivationsBase):
    """Residual stream activations produced by a single ViT residual block."""

    residual_pre: torch.Tensor | InterventionProxy = field(default_factory=empty_tensor)
    residual_mid: torch.Tensor | InterventionProxy = field(default_factory=empty_tensor)
    residual_post: torch.Tensor | InterventionProxy = field(
        default_factory=empty_tensor
    )


@dataclass
class BlockAttention(BlockActivationsBase):
    """Attention patterns produced by a single ViT attention layer."""

    attn_pattern: torch.Tensor | InterventionProxy = field(default_factory=empty_tensor)
