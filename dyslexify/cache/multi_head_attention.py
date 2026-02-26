# mypy: allow-untyped-defs
from typing import Optional

import einops
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


class MultiheadAttentionWithWeightHook(MultiheadAttention):
    r"""Multi-head attention with support for attention pattern and result manipulation hooks.

    This is a simplified version of MultiheadAttention that always uses custom attention
    computation to support hooks for manipulating both attention patterns and attention results.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O

    where :math:`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    This implementation always uses custom attention computation to enable hook intervention
    for attention pattern and result manipulation.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = MultiheadAttentionWithWeightHook(embed_dim, num_heads)
        >>>
        >>> # Register a hook to manipulate attention patterns
        >>> def pattern_hook(attn_weights, query, key, value):
        >>>     # Modify attention weights as needed
        >>>     return modified_attn_weights
        >>> multihead_attn.register_attention_pattern_hook(pattern_hook)
        >>>
        >>> # Register a hook to manipulate attention results
        >>> def result_hook(attn_output, attn_weights, query, key, value):
        >>>     # Modify attention output as needed
        >>>     return modified_attn_output
        >>> multihead_attn.register_attn_result_hook(result_hook)
        >>>
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    """

    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_pattern_hooks = (
            []
        )  # Hook function for attention pattern manipulation
        self.attn_result_hooks = []  # Hook function for attention result manipulation

    def register_attention_pattern_hook(self, hook_fn):
        """Register a hook function that will be called with the attention weights.

        Args:
            hook_fn: Function that takes attention weights and returns modified weights.
                     Signature: hook_fn(attn_weights, query, key, value) -> modified_attn_weights
        """
        self.attention_pattern_hooks.append(hook_fn)

    def register_attn_result_hook(self, hook_fn):
        """Register a hook function that will be called with the attention result.

        Args:
            hook_fn: Function that takes attention_ result and returns modified result.
                     Signature: hook_fn(attn_output, attn_weights, query, key, value) -> modified_attn_output
        """
        self.attn_result_hooks.append(hook_fn)

    def remove_attention_pattern_hook(self):
        """Remove the attention pattern hook."""
        self.attention_pattern_hooks = []

    def remove_attn_result_hook(self):
        """Remove the attention result hook."""
        self.attn_result_hooks = []

    def remove_all_hooks(self):
        """Remove all hooks."""
        self.attention_pattern_hooks = []
        self.attn_result_hooks = []

    def _compute_attention_with_hook(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
        average_attn_weights=True,
        is_causal=False,
    ):
        """Compute attention with hook intervention for pattern manipulation."""
        # Project Q, K, V (always same embed dim)
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(
            3, dim=-1
        )

        # Determine input format and reshape accordingly
        if self.batch_first:
            # Input is in (batch, seq_len, embed_dim) format
            batch_size = q.size(0)
            seq_len = q.size(1)
            head_dim = self.embed_dim // self.num_heads

            # Reshape directly to (batch, num_heads, seq_len, head_dim)
            q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        else:
            # Input is in (seq_len, batch, embed_dim) format
            batch_size = q.size(1)  # batch is at dim 1
            seq_len = q.size(0)  # seq_len is at dim 0
            head_dim = self.embed_dim // self.num_heads

            q = einops.rearrange(q, "s b (h d) -> b h s d", h=self.num_heads)
            k = einops.rearrange(k, "s b (h d) -> b h s d", h=self.num_heads)
            v = einops.rearrange(v, "s b (h d) -> b h s d", h=self.num_heads)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Reshape key_padding_mask to match attention weights
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        # Apply hook if registered
        for hook in self.attention_pattern_hooks:
            attn_weights = hook(attn_weights, q, k, v)

        # Apply attention weights to values
        attn_output = torch.matmul(
            attn_weights, v
        )  # [batch, num_heads, seq_len, head_dim]

        # Apply attention result hook if registered
        for hook in self.attn_result_hooks:
            attn_output = hook(attn_output, attn_weights, q, k, v)

        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        # Apply output projection
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        # Prepare return values
        if need_weights:
            if average_attn_weights:
                attn_output_weights = attn_weights.mean(dim=1)  # Average across heads
            else:
                attn_output_weights = attn_weights
        else:
            attn_output_weights = None

        return attn_output, attn_output_weights

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        r"""Compute attention outputs using query, key, and value embeddings.

        Simplified version that always uses custom attention computation with hook support.
        """
        # Store original input format
        original_query_shape = query.shape
        is_batched = query.dim() == 3

        # Handle batch_first format - only transpose if we need to convert to (seq_len, batch, embed_dim)
        # Since OpenCLIP already passes input in batch_first format, we don't need to transpose
        if not self.batch_first and is_batched:
            # If not batch_first, we need to transpose to (seq_len, batch, embed_dim) for processing
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # Always use custom attention computation with hook support
        attn_output, attn_output_weights = self._compute_attention_with_hook(
            query,
            key,
            value,
            key_padding_mask,
            attn_mask,
            need_weights,
            average_attn_weights,
            is_causal,
        )

        # Ensure output matches input format
        if not self.batch_first and is_batched:
            # If input was not batch_first, transpose output back
            attn_output = attn_output.transpose(1, 0)
            if attn_output_weights is not None:
                attn_output_weights = attn_output_weights.transpose(1, 0)

        return attn_output, attn_output_weights
