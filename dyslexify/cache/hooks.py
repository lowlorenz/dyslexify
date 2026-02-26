# %%

import torch
from typing import Optional, Callable, Dict, Any


def create_zero_cls_to_spatial_hook(head_idx: int) -> Callable:
    """
    Create an attention pattern hook that zeros out CLS attention to spatial tokens for a specific layer and head.

    Args:
        layer_idx: The layer index (0-11 for ViT-B/32)
        head_idx: The attention head index (0-11 for ViT-B/32)

    Returns:
        An attention pattern hook function that can be registered with register_attention_pattern_hook
    """

    def zero_cls_to_spatial_hook(
        attn_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out CLS attention to spatial tokens for the specified head.

        Args:
            attn_weights: Attention weights tensor of shape [batch, num_heads, seq_len, seq_len]
            query: Query tensor
            key: Key tensor
            value: Value tensor

        Returns:
            Modified attention weights tensor
        """
        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
        # CLS token is at position 0, spatial tokens are at positions 1-196 (for 14x14 patches)

        # Create a copy to avoid modifying the original tensor
        modified_weights = attn_weights.clone()

        # Directly set CLS attention to [1, 0, 0, 0, ...] for the specified head
        # This makes the CLS token attend only to itself
        modified_weights[:, head_idx, 0, 0] = 1.0  # CLS to CLS = 1
        modified_weights[:, head_idx, 0, 1:] = 0.0  # CLS to spatial = 0

        return modified_weights

    return zero_cls_to_spatial_hook


def create_zero_cls_attention_result_hook(head_idx: int) -> Callable:
    """
    Create a hook function that zeros out the attention output for a specific head.

    This hook is used to ablate (disable) specific attention heads during experiments
    by setting their output to zero.

    Args:
        head_idx (int): Index of the attention head to zero out

    Returns:
        Callable: Hook function that takes attention output, weights, and key/value tensors
                 and returns modified attention output with the specified head zeroed
    """

    def zero_cls_attention_result_hook(attn_output, attn_weights, q, k, v):
        attn_output[:, head_idx, 0, :] = 0
        return attn_output

    return zero_cls_attention_result_hook


def create_zero_spatial_attention_result_hook(head_idx: int) -> Callable:
    """
    Create a hook function that zeros out the attention output of the spatial tokens for a specific head

    Args:
        head_idx (int): Index of the attention head to zero out

    Returns:
        Callable: Hook function that takes attention output, weights, and key/value tensors
                 and returns modified attention output with the specified head zeroed
    """

    def zero_spatial_attention_result_hook(attn_output, attn_weights, q, k, v):
        attn_output[:, head_idx, 1:, :] = 0
        return attn_output

    return zero_spatial_attention_result_hook


def create_head_ablation_hook(layer_idx: int, head_idx: int) -> Callable:
    """
    Create an attention pattern hook that zeros out a specific head.
    """

    def head_ablation_hook(
        attn_output: torch.Tensor,
        attn_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out a specific head.

        Args:
            attn_weights: Attention weights tensor of shape [batch, num_heads, seq_len, seq_len]
            query: Query tensor
            key: Key tensor
            value: Value tensor

        Returns:
            Modified attention weights tensor
        """
        modified_output = attn_output.clone()

        # Zero out the specified head
        modified_output[:, head_idx, :, :] = 0.0

        return modified_output

    return head_ablation_hook


def create_attention_capture_hook(collector_instance) -> Callable:
    """
    Create an attention pattern capture hook that stores patterns in the collector.

    Args:
        collector_instance: The OpenClipActivationCollector instance to store patterns in.

    Returns:
        A hook function that captures attention patterns and stores them in the collector.
    """

    def attention_capture_hook(attn_weights, query, key, value):
        """Capture attention patterns during forward pass."""
        # This hook will be called for each layer, but we need to know which layer
        # We'll use a closure to capture the layer index
        return attn_weights

    return attention_capture_hook


def create_layer_specific_attn_capture_hook(
    layer_idx: int, cls_token_only: bool = False
) -> Callable:
    """
    Create an attention pattern capture hook for a specific layer.

    Args:
        layer_idx: The index of the layer to capture attention patterns for.

    Returns:
        A hook function that captures attention patterns for the specified layer.
        The hook has a 'captured_patterns' property that stores the captured patterns.
    """

    def attention_capture_hook(attn_weights, query, key, value):
        """Capture attention patterns during forward pass."""
        # Store the attention weights for this layer
        if cls_token_only:
            attention_capture_hook.captured_patterns[layer_idx] = (
                attn_weights.detach().clone()[:, :, 0, :]
            )
        else:
            attention_capture_hook.captured_patterns[layer_idx] = (
                attn_weights.detach().clone()
            )
        # Return the original weights unchanged
        return attn_weights

    # Initialize the captured_patterns property
    attention_capture_hook.captured_patterns = {}

    return attention_capture_hook


def create_hook_factory() -> Dict[str, Callable]:
    """
    Create a factory of pre-configured hooks.

    Returns:
        Dictionary mapping hook names to hook creation functions
    """
    return {
        # Attention pattern hooks
        "zero_cls_to_spatial": create_zero_cls_to_spatial_hook,
        # Attention capture hooks
        "attention_capture": create_attention_capture_hook,
        "layer_specific_capture": create_layer_specific_attn_capture_hook,
    }


# Convenience function for easy hook creation
def get_hook(hook_type: str, **kwargs) -> Callable:
    """
    Get a hook function by type with specified parameters.

    Args:
        hook_type: Type of hook to create
        **kwargs: Parameters for the hook creation function

    Returns:
        A hook function that can be registered with MultiheadAttentionWithWeightHook

    Examples:
        # Zero out CLS attention to spatial tokens for layer 3, head 5
        pattern_hook = get_hook("zero_cls_to_spatial", layer_idx=3, head_idx=5)
        attention_layer.register_attention_pattern_hook(pattern_hook)

        # Amplify CLS token contribution for layer 2, head 7
        result_hook = get_hook("cls_token_amplifier", layer_idx=2, head_idx=7, amplification_factor=3.0)
        attention_layer.register_attn_result_hook(result_hook)
    """
    factory = create_hook_factory()

    if hook_type not in factory:
        available_hooks = list(factory.keys())
        raise ValueError(
            f"Unknown hook type '{hook_type}'. Available hooks: {available_hooks}"
        )

    hook_creator = factory[hook_type]
    return hook_creator(**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test attention pattern hook
    print("=== Testing Attention Pattern Hook ===")
    pattern_hook = get_hook("zero_cls_to_spatial", layer_idx=3, head_idx=5)

    # Test the hook with dummy data
    batch_size = 2
    num_heads = 12
    seq_len = 197  # CLS + 196 spatial tokens

    # Create dummy attention weights
    attn_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    print("Original CLS attention weights for head 5:")
    print(f"CLS to CLS: {attn_weights[0, 5, 0, 0]:.6f}")
    print(f"CLS to spatial (first 5): {attn_weights[0, 5, 0, 1:6]}")
    print(f"CLS to spatial (last 5): {attn_weights[0, 5, 0, -5:]}")

    # Apply the pattern hook
    modified_weights = pattern_hook(attn_weights, None, None, None)

    print("\nModified CLS attention weights for head 5:")
    print(f"CLS to CLS: {modified_weights[0, 5, 0, 0]:.6f}")
    print(f"CLS to spatial (first 5): {modified_weights[0, 5, 0, 1:6]}")
    print(f"CLS to spatial (last 5): {modified_weights[0, 5, 0, -5:]}")

    # Test attention capture hook
    print("\n=== Testing Attention Capture Hook ===")

    capture_hook = create_layer_specific_attn_capture_hook(layer_idx=0)

    # Apply the capture hook
    captured_weights = capture_hook(attn_weights, None, None, None)

    print(f"Original attention weights shape: {attn_weights.shape}")
    print(f"Captured attention weights shape: {captured_weights.shape}")
    print(f"Patterns stored in hook: {list(capture_hook.captured_patterns.keys())}")

    # Verify the hooks worked correctly
    cls_to_spatial_zeroed = torch.allclose(
        modified_weights[0, 5, 0, 1:197], torch.zeros(196), atol=1e-6
    )
    cls_to_cls_preserved = modified_weights[0, 5, 0, 0] > 0
    capture_working = 0 in capture_hook.captured_patterns

    print(f"\nVerification:")
    print(f"CLS attention to spatial tokens zeroed: {cls_to_spatial_zeroed}")
    print(f"CLS attention to CLS token preserved: {cls_to_cls_preserved}")
    print(f"Attention capture working: {capture_working}")

    print(
        f"\nHook tests {'PASSED' if cls_to_spatial_zeroed and cls_to_cls_preserved and capture_working else 'FAILED'}"
    )
