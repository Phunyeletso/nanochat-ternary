"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on incompatible CUDA GPUs, MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
# STUDY NOTE (added): attention math is  Y = softmax(Q K^T / sqrt(D) + mask) V.
# Q K^T is a (T x T) matrix of similarity scores (every token vs every token) -> memory and FLOPs grow as T^2.
# "Flash" attention computes the exact same result in tiles that fit in fast on-chip GPU memory (SRAM),
# never writing the full T x T matrix to slow GPU memory (HBM). Same math, far less memory traffic.

# Import PyTorch
import torch
# Import the functional API (provides scaled_dot_product_attention = SDPA)
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
# Try to load the fast Flash Attention 3 GPU kernel
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    # No NVIDIA GPU available...
    if not torch.cuda.is_available():
        # ...so FA3 can't be used
        return None
    # Any failure below means "not available"
    try:
        # Ask the GPU for its architecture generation (e.g. 9 = Hopper/H100)
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        # Import os for environment variables
        import os
        # Silence download progress bars
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Import helpers that download precompiled GPU kernels from the Hugging Face hub
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        # Hopper GPUs (H100)
        if major == 9:
            # Use this kernel repo
            hf_kernel = "varunneal/flash-attention-3"
            # Download/load and return its interface
            return get_kernel(hf_kernel).flash_attn_interface
        # Other GPUs
        else:
            # Use the community build
            hf_kernel = "kernels-community/flash-attn3"
            # If a build exists for this GPU...
            if has_kernel(hf_kernel):
                # ...load and return it
                return get_kernel(hf_kernel).flash_attn_interface
            # Otherwise...
            else:
                # ...not available
                return None

    # If anything went wrong...
    except Exception:
        # ...fall back
        return None


# Try loading once at import time
_fa3 = _load_flash_attention_3()
# Flag: is FA3 available?
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
# Testing override (None = decide automatically)
_override_impl = None


# Decide which implementation to use
def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    # Forced FA3
    if _override_impl == 'fa3':
        # Must actually be available
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        # Use FA3
        return True
    # Forced SDPA
    if _override_impl == 'sdpa':
        # Don't use FA3
        return False
    # FA3 is available
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        # Check the compute dtype
        from nanochat.common import COMPUTE_DTYPE
        # bf16 is supported...
        if COMPUTE_DTYPE == torch.bfloat16:
            # ...so use FA3
            return True
        # Other dtypes aren't
        return False
    # FA3 not available
    return False

# Make the decision once
USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
# Attention using PyTorch's built-in scaled_dot_product_attention, with sliding-window support
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    # Number of query positions
    Tq = q.size(2)
    # Number of key positions
    Tk = k.size(2)
    # How far back each query may look
    window = window_size[0]

    # Full context, same length
    # Simple case: no effective window and queries/keys aligned
    if (window < 0 or window >= Tq) and Tq == Tk:
        # Standard causal attention: softmax(QK^T/sqrt(D) + lower-triangular mask) V
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    # Generating one token at a time
    if Tq == 1:
        # If the window is smaller than the cache...
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            # First key index inside the window
            start = max(0, Tk - (window + 1))
            # Keep only the keys in the window
            k = k[:, :, start:, :]
            # Keep only the values in the window
            v = v[:, :, start:, :]
        # One query attends to all (remaining) keys; no mask needed
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    # Device for the mask
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    # Absolute position of each query (column vector)
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    # Absolute position of each key (row vector)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    # Causal: query may see key only if key position <= query position
    mask = col_idx <= row_idx

    # sliding window (left)
    # If a sliding window applies...
    if window >= 0 and window < Tk:
        # ...also require the key to be within `window` positions
        mask = mask & ((row_idx - col_idx) <= window)

    # Attention with the explicit boolean mask
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
# Training-time attention
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    # Fast path
    if USE_FA3:
        # Call FA3 directly
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    # SDPA wants heads before time
    q = q.transpose(1, 2)
    # Same for k
    k = k.transpose(1, 2)
    # Same for v
    v = v.transpose(1, 2)
    # Fewer KV heads than Q heads means grouped-query attention
    enable_gqa = q.size(1) != k.size(1)
    # Run attention
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    # Transpose back to (B, T, H, D)
    return y.transpose(1, 2)  # back to (B, T, H, D)


# Inference-time attention with a key/value cache
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            # Masking options
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    # Fast path
    if USE_FA3:
        # FA3 handles cache insertion and attention in one kernel
        return _fa3.flash_attn_with_kvcache(
            # Inputs
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            # Options
            causal=causal, window_size=window_size
        # Close call
        )

    # SDPA fallback: manually manage KV cache
    # Shapes of the new queries
    B, T_new, H, D = q.shape
    # Current write position in the cache
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    # If new keys/values were given...
    if k is not None and v is not None:
        # ...write keys into the cache
        k_cache[:, pos:pos+T_new, :, :] = k
        # ...write values into the cache
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    # End of valid cache region
    end_pos = pos + T_new
    # All keys so far
    k_full = k_cache[:, :end_pos, :, :]
    # All values so far
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    # Queries
    q_sdpa = q.transpose(1, 2)
    # Keys
    k_sdpa = k_full.transpose(1, 2)
    # Values
    v_sdpa = v_full.transpose(1, 2)

    # GQA if head counts differ
    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    # Run attention
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    # Back to (B, T, H, D)
    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
# SimpleNamespace = an object you can attach attributes to
from types import SimpleNamespace
# Bundle the two functions so callers write flash_attn.flash_attn_func(...)
flash_attn = SimpleNamespace(
    # Training function
    flash_attn_func=flash_attn_func,
    # Inference function
    flash_attn_with_kvcache=flash_attn_with_kvcache,
# Close
)
