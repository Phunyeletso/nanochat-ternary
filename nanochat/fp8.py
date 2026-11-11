"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
subclass dispatch tables, etc.)

How FP8 training works
======================
A standard Linear layer does one matmul in forward and two in backward:
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
  4. Dequantize: _scaled_mm handles this internally using the inverse scales

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
torchao is just orchestration around these primitives. We can call them directly.

FP8 dtype choice
================
There are two FP8 formats. We use both, following the standard convention:
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.

torch._scaled_mm layout requirements
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
  - First argument (A):  must be row-major (contiguous)
  - Second argument (B): must be column-major (B.t().contiguous().t())
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
already column-major — no copy needed. Otherwise we use _to_col_major().

How this differs from torchao's approach
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
  output = input @ weight.T
the @ operator dispatches to aten.mm, which gets intercepted and routed to
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
a handler for every tensor operation that might touch an FP8 tensor.

We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
opaque node rather than trying to trace inside.

The trade-off is in how torch.compile sees the two approaches:
  - torchao: compile decomposes the tensor subclass (via __tensor_flatten__) and
    sees every individual op (amax, scale, cast, _scaled_mm) as separate graph
    nodes. Inductor can fuse these with surrounding operations (e.g. fuse the
    amax computation with the preceding layer's activation function).
  - ours: compile sees a single opaque call. It can optimize everything around
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
compared to the matmul. In practice this means our version is slightly faster
(less compilation overhead, no tensor subclass dispatch cost) but can produce
subtly different floating-point rounding paths under torch.compile, since Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
"""
# STUDY NOTE (added): this file is the clearest place to SEE the calculus of a linear layer.
# If Y = X W^T and the loss gradient wrt Y is G = dL/dY, the chain rule gives two more matmuls:
#   dL/dX = G W        (passed backwards to earlier layers)
#   dL/dW = G^T X      (used by the optimizer to update W)
# So backward = 2 matmuls, forward = 1 matmul  =>  total cost ~3x the forward pass. That is the "6N" rule:
# 2 FLOPs per param forward + 4 FLOPs per param backward = 6 FLOPs per parameter per training token.
# Fewer bits per number (8 instead of 16) => GPU tensor cores do ~2x more multiplies per second.

# Import PyTorch
import torch
# Import neural-network layers
import torch.nn as nn

# Import the compute dtype (bf16)
from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
# Tiny constant to avoid divide-by-zero
EPS = 1e-12


# Quantization doesn't need gradients
@torch.no_grad()
# Convert a tensor to 8-bit float with a single scale factor
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    """
    # Largest number representable in this FP8 format (448 or 57344)
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    # Largest magnitude in the tensor
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # ensure consistent numerics between torch.compile and eager mode.
    # (torchao does the same upcast — without it, compile/eager can diverge)
    # Scale factor that stretches the tensor to fill the FP8 range (uses all available precision)
    scale = fp8_max / amax.double().clamp(min=EPS)
    # Back to fp32
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    # Multiply by the scale
    x_scaled = x.float() * scale
    # Clamp to the representable range
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    # Round to 8-bit float
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # convert FP8 values back to the original range during the matmul)
    # 1/scale, used to undo the scaling after the matmul
    inv_scale = scale.reciprocal()
    # Return quantized data and inverse scale
    return x_fp8, inv_scale


# Change memory layout to column-major (what the FP8 kernel needs for its 2nd operand)
def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.

    torch._scaled_mm requires its second operand in column-major layout.
    The trick: transpose -> contiguous (forces a copy in transposed order)
    -> transpose back. The result has the same logical shape but column-major
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    """
    # Same values, different storage order
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# for how this differs from torchao's tensor subclass approach.
# Let torch.compile treat this as one black-box op
@torch._dynamo.allow_in_graph
# A custom autograd function: we write forward AND backward (the derivative) by hand
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.

    The forward quantizes input and weight to FP8 and saves
    the quantized tensors + scales for backward.
    """

    # No `self` needed
    @staticmethod
    # Forward: compute Y = X W^T
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        # Quantize input X
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        # Quantize weight W
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        # Save them: backward needs X and W to compute the gradients
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        # FP8 matmul: [B,K] @ [K,N] -> [B,N]
        output = torch._scaled_mm(
            # A = X
            input_fp8,
            # B = W^T
            weight_fp8.t(),
            # Undo X's scale
            scale_a=input_inv,
            # Undo W's scale
            scale_b=weight_inv,
            # Output dtype (bf16)
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # Slightly less accurate but measurably faster. Standard practice for
            # the forward pass; we use False in backward for more precise gradients.
            # Faster, slightly less precise summation
            use_fast_accum=True,
        # Close call
        )
        # Return Y
        return output

    # No `self` needed
    @staticmethod
    # Backward: given G = dL/dY, return dL/dX and dL/dW (chain rule)
    def backward(ctx, grad_output):
        # Retrieve the saved X and W
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        # Quantize G with the wide-range format
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        # Rearrange W's memory
        w_col = _to_col_major(w_fp8)
        # dL/dX = G @ W  (because Y = X W^T, so dY/dX contributes W)
        grad_input = torch._scaled_mm(
            # A = G
            go_fp8,
            # B = W
            w_col,
            # Undo G's scale
            scale_a=go_inv,
            # Undo W's scale
            scale_b=w_inv,
            # Output dtype
            out_dtype=grad_output.dtype,
            # Precise summation for gradients
            use_fast_accum=False,
        # Close call
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # Transposing gives column-major, but first arg needs row-major,
        # so we must call .contiguous() to physically rearrange the memory.
        # G^T in row-major layout
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
        # X in column-major layout
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
        # dL/dW = G^T @ X  (sum over the batch of outer products: each token contributes g_t x_t^T)
        grad_weight = torch._scaled_mm(
            # A = G^T
            go_T,
            # B = X
            in_col,
            # Undo G's scale
            scale_a=go_inv,
            # Undo X's scale
            scale_b=in_inv,
            # Output dtype
            out_dtype=grad_output.dtype,
            # Precise summation
            use_fast_accum=False,
        # Close call
        )

        # Return gradients wrt each forward input (X, W)
        return grad_input, grad_weight


# A Linear layer whose matmuls run in FP8
class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    """

    # Forward pass
    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # reduced precision input, and we no longer rely on autocast to do this.
        # Make sure input is bf16
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        # Remember original shape (B, T, C)
        orig_shape = input.shape
        # Flatten to (B*T, C): a matmul only cares about rows, so batch and time merge
        input_2d = input.reshape(-1, orig_shape[-1])
        # Run the FP8 matmul (with custom backward)
        output = _Float8Matmul.apply(input_2d, self.weight)
        # Restore (B, T, out_features)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        # If the layer has a bias...
        if self.bias is not None:
            # ...add it
            output = output + self.bias.to(output.dtype)
        # Return result
        return output

    # Alternative constructor
    @classmethod
    # Build a Float8Linear sharing the weights of an existing nn.Linear
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.

        Uses meta device to avoid allocating a temporary weight tensor — we
        create the module shell on meta (shapes/dtypes only, no memory), then
        point .weight and .bias to the original module's parameters.
        """
        # On the "meta" device, tensors have shapes but no memory
        with torch.device("meta"):
            # Create the empty shell
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        # Point to the original weight (no copy)
        new_mod.weight = mod.weight
        # Point to the original bias
        new_mod.bias = mod.bias
        # Return the new module
        return new_mod


# Minimal config object (API compatibility with torchao)
class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""

    # No `self`
    @staticmethod
    # Build config from a recipe name
    def from_recipe_name(recipe_name):
        # Only "tensorwise" is supported
        if recipe_name != "tensorwise":
            # Otherwise error out
            raise ValueError(
                # Message part 1
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                # Message part 2
                f"Rowwise/axiswise recipes require the full torchao library."
            # Close
            )
        # Return a config instance
        return Float8LinearConfig()


# Swap every nn.Linear in a model for a Float8Linear
def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.

    Walks the module tree in post-order (children before parents) and swaps
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    the original weight and bias tensors — no copies, no extra memory.

    Args:
        module: Root module to convert.
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
            are converted. Common use: skip layers with dims not divisible by 16
            (hardware requirement for FP8 matmuls on H100).
    """
    # Recursive helper
    def _convert(mod, prefix=""):
        # For every direct child module
        for name, child in mod.named_children():
            # Its fully-qualified name, e.g. "transformer.h.0.attn.c_q"
            fqn = f"{prefix}.{name}" if prefix else name
            # Recurse into the child first
            _convert(child, fqn)
            # If it's a regular Linear (not already converted)...
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                # ...and passes the filter...
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    # ...replace it
                    setattr(mod, name, Float8Linear.from_float(child))

    # Start the recursion
    _convert(module)
    # Return the (modified in-place) module
    return module
