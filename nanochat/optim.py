"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
The same class handles both single GPU and distributed training: when there is no
multi-rank process group, the communication ops are simply skipped and every rank
(i.e. the only rank) owns all of the parameters.

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""
# =====================================================================================
# STUDY NOTES (added): what an optimizer does (the calculus part of training)
#   1. Forward pass computes a scalar loss L(W) from the weights W.
#   2. loss.backward() uses the chain rule to compute the gradient g = dL/dW for every weight.
#      The gradient points in the direction where the loss INCREASES fastest.
#   3. The optimizer moves each weight a small step AGAINST the gradient: W <- W - lr * update(g).
#      Plain gradient descent uses update(g) = g. Adam and Muon use smarter versions of g.
#   - AdamW: per-number adaptive step = momentum / sqrt(average of g^2)  (elementwise, no matrix math)
#   - Muon: treats each weight as a MATRIX and replaces the update with its nearest orthogonal matrix
#           (all singular values set to ~1), computed with a few matrix multiplications (Newton-Schulz).
# =====================================================================================

# Import PyTorch
import torch
# Import distributed communication (sending gradients between GPUs)
import torch.distributed as dist
# Import the Tensor type for type hints
from torch import Tensor
# Import the dtype used for math (bf16 normally)
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
https://arxiv.org/abs/1711.05101
"""

# Compile this function into a single fused GPU kernel (fewer memory round-trips = faster)
@torch.compile(dynamic=False, fullgraph=True)
# One AdamW update for one parameter tensor
def adamw_step_fused(
    # The weights being updated
    p: Tensor,              # (32768, 768) - parameter tensor
    # dL/dp, computed by backprop
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
    # m: running average of gradients (momentum, "which direction have we been going")
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
    # v: running average of squared gradients ("how big/noisy is this coordinate's gradient")
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
    # Step counter t
    step_t: Tensor,         # () - 0-D CPU tensor, step count
    # Learning rate (step size)
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
    # Decay rate for m
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
    # Decay rate for v
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
    # Tiny number to avoid division by zero
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
    # Weight decay strength (pulls weights towards zero = regularization)
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
# Returns nothing (updates in place)
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    All in one compiled graph to eliminate Python overhead between ops.
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    """
    # Some params (wte, value_embeds) are stored in bf16, so do the math in fp32 and
    # cast back at the end. MPS errors on mixed-dtype ops (CUDA promotes them), and
    # scalar arithmetic like 1 - beta2 loses all precision in bf16. compile fuses the casts.
    # fp32 copy of the weights
    p32 = p.float()
    # fp32 copy of m
    exp_avg32 = exp_avg.float()
    # fp32 copy of v
    exp_avg_sq32 = exp_avg_sq.float()
    # fp32 copy of the gradient
    grad32 = grad.float()
    # Weight decay (decoupled, applied before the update)
    # Shrink weights: p <- p * (1 - lr*wd)
    p32.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    # m <- beta1*m + (1-beta1)*g   (exponential moving average of the gradient)
    exp_avg32.lerp_(grad32, 1 - beta1_t)
    # v <- beta2*v + (1-beta2)*g^2 (exponential moving average of the squared gradient)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2_t)
    # Bias corrections
    # m starts at 0 so early averages are too small; divide by (1 - beta1^t) to correct
    bias1 = 1 - beta1_t ** step_t
    # Same correction for v
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    # Denominator sqrt(v_hat) + eps: coordinates with big/noisy gradients take smaller steps
    denom = (exp_avg_sq32 / bias2).sqrt() + eps_t
    # Effective step size including bias correction
    step_size = lr_t / bias1
    # p <- p - lr * m_hat / (sqrt(v_hat) + eps). Each element gets its own adaptive step.
    p32.add_(exp_avg32 / denom, alpha=-step_size)
    # Write back (no-ops in the common case where everything is already fp32)
    # Write updated weights back to the original tensor
    p.copy_(p32)
    # Write m back
    exp_avg.copy_(exp_avg32)
    # Write v back
    exp_avg_sq.copy_(exp_avg_sq32)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Two more (very) slight and optional improvements:
1) MuonEq row equilibration: rescale each row to the mean row norm so the spectrum
entering orthogonalization is better conditioned (https://arxiv.org/abs/2603.28254)
2) Muon+ renormalization: snap the Frobenius norm to sqrt(min(m, n)), the norm of an exactly
semi-orthogonal matrix, correcting for under-convergence of the polar iteration (https://arxiv.org/abs/2602.21545)

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""
# LINEAR ALGEBRA NOTE (added): Any matrix G has an SVD  G = U S V^T  (rotate, stretch by singular values S, rotate).
# The gradient matrix is usually dominated by a few huge singular values, so plain SGD mostly moves weights
# in a few directions. Muon replaces G with U V^T (all singular values -> 1), so every direction gets an
# equal-sized step. Computing an SVD is slow on GPUs, so instead we apply a polynomial p(X) = aX + b(XX^T)X + c(XX^T)^2 X
# a few times; each application pushes every singular value towards 1 using only matmuls (which GPUs are great at).

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
# The (a, b, c) polynomial coefficients for each of the 5 iterations
polar_express_coeffs = [
    # Iteration 1: very aggressive, pushes small singular values up fast
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    # Iteration 2
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    # Iteration 3
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    # Iteration 4
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    # Iteration 5: gentler, settles values near 1
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
# Close the list
]


# Compile into one fused GPU graph
@torch.compile(dynamic=False, fullgraph=True)
# One Muon update for a stack of same-shaped weight matrices
def muon_step_fused(
    # K gradient matrices stacked into one 3D tensor
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
    # The K weight matrices being updated
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
    # Momentum (running average of gradients) per matrix
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
    # Per-row or per-column running average of squared updates (much smaller than a full matrix)
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
    # Momentum coefficient
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
    # Learning rate
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
    # Weight decay
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
    # Decay rate for the second moment
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
    # Number of orthogonalization iterations
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
    # Which dimension (rows or columns) to compute variance over
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
# Returns nothing
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    """

    # Nesterov momentum
    # Momentum coefficient in the right dtype
    momentum = momentum_t.to(stacked_grads.dtype)
    # buf <- momentum*buf + (1-momentum)*grad (moving average of gradients: smooths noisy minibatch gradients)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    # Nesterov "look-ahead": g <- (1-momentum)*grad + momentum*buf
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    # X is the matrix we'll orthogonalize; bf16 matmuls are much faster
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration: rescale each row to the mean row norm so the spectrum entering orthogonalization is better conditioned
    # Target row norm = (Frobenius norm) / sqrt(number of rows) = average row length
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    # Current length of each row (clamped to avoid dividing by 0)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # Rescale every row to the average length (better "conditioning" = singular values less spread out)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization: replace each update with the nearest orthogonal matrix
    # Divide by Frobenius norm so all singular values are <= 1 (the polynomial only works on [0,1])
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    # More rows than columns: work with the smaller (cols x cols) Gram matrix X^T X to save FLOPs
    if g.size(-2) > g.size(-1): # Tall matrix
        # Run each iteration with its coefficients
        for a, b, c in polar_express_coeffs[:ns_steps]:
            # Gram matrix A = X^T X (its eigenvalues are the squared singular values of X)
            A = X.mT @ X
            # B = bA + cA^2 (a polynomial in A)
            B = b * A + c * (A @ A)
            # X <- aX + X B = X(aI + bA + cA^2): maps each singular value s to a*s + b*s^3 + c*s^5
            X = a * X + X @ B
    # Wide (or square) matrix: use the (rows x rows) Gram matrix X X^T
    else: # Wide matrix (original math)
        # Run each iteration
        for a, b, c in polar_express_coeffs[:ns_steps]:
            # Gram matrix A = X X^T
            A = X @ X.mT
            # Polynomial in A
            B = b * A + c * (A @ A)
            # X <- aX + B X, same singular-value polynomial as above
            X = a * X + B @ X
    # Cast back to the param dtype (MPS errors on the mixed-dtype ops below when X is bf16)
    # Back to the parameters' dtype
    g = X.to(stacked_params.dtype)

    # Muon+ renormalization: snap Frobenius norm to sqrt(min(m, n))
    # A perfectly orthogonal m x n matrix has min(m,n) singular values of 1 => Frobenius norm sqrt(min(m,n))
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    # Current Frobenius norm
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    # Rescale to the ideal norm
    g = g * (target_norm / current_norm).to(g.dtype)

    # Variance reduction
    # beta2 in the right dtype
    beta2 = beta2_t.to(g.dtype)
    # Mean squared value per row (or per column): how big each neuron's update is
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    # Length of the dimension we averaged over
    red_dim_size = g.size(red_dim)
    # Total squared Frobenius norm of the update
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    # Frobenius norm of the update before rescaling
    v_norm = v_norm_sq.sqrt()
    # Running average of per-neuron squared update size (like Adam's v, but one number per row/col)
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    # Per-neuron scale = 1/sqrt(running average): neurons with big updates get scaled down
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    # Squared norm after applying the per-neuron scales
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    # Norm after rescaling
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    # Final scale: per-neuron rebalancing, but keep the overall norm the same as before
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    # Apply the scale
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    # Learning rate in the right dtype
    lr = lr_t.to(g.dtype)
    # Weight decay in the right dtype
    wd = wd_t.to(g.dtype)
    # Only decay weights where the update already pushes them towards zero (same sign as the weight)
    mask = (g * stacked_params) >= 0
    # W <- W - lr*update - lr*wd*W*mask  (the actual gradient-descent step)
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------

# The combined optimizer class, inheriting PyTorch's Optimizer base class
class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    The same class covers single GPU and distributed training. In the distributed setting
    (a multi-rank process group is initialized), gradients are synchronized here in the
    optimizer (nanochat does not use DDP) and optimizer states are sharded across ranks
    (ZeRO-2 style). On a single rank, all communication is skipped and the rank owns all
    parameters, so the sharded code paths degenerate to plain full-tensor updates.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    # Constructor
    def __init__(self, param_groups: list[dict]):
        # Let the base Optimizer class store the parameter groups
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # AdamW step-counter holder
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # AdamW learning-rate holder
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # AdamW beta1 holder
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # AdamW beta2 holder
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # AdamW epsilon holder
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # AdamW weight-decay holder
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon momentum holder
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon learning-rate holder
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon weight-decay holder
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon beta2 holder
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    # Phase 1 for AdamW: start averaging gradients across GPUs
    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        # Per-parameter bookkeeping
        param_infos = {}
        # For each parameter in the group
        for p in group['params']:
            # Its gradient from backprop
            grad = p.grad
            # Only one GPU
            if world_size == 1:
                # Single rank: no communication, update the full param in place
                # Nothing to communicate: use the whole gradient
                param_infos[p] = dict(future=None, grad_slice=grad, is_small=True)
            # Tiny parameter
            elif p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                # Average the gradient across all GPUs (each GPU saw different data)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                # Remember the pending operation
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            # Large parameter
            else:
                # Large params: reduce_scatter
                # Rows must split evenly across GPUs
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                # Rows per GPU
                rank_size = grad.shape[0] // world_size
                # Buffer for this GPU's slice of the averaged gradient
                grad_slice = torch.empty_like(grad[:rank_size])
                # Average across GPUs AND give each GPU only its slice (saves memory)
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                # Remember the pending operation
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        # Return bookkeeping
        return dict(param_infos=param_infos)

    # Phase 1 for Muon: stack gradients and start averaging across GPUs
    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        # Parameters in this group (all same shape)
        params = group['params']
        # Single GPU
        if world_size == 1:
            # Single rank: this rank owns all params, the stacked grads are the "chunk"
            # Stack all gradients into one (K, rows, cols) tensor
            grad_chunk = torch.stack([p.grad for p in params])
            # Return without communication
            return dict(future=None, grad_chunk=grad_chunk, stacked_grads=None, chunk_size=len(params))
        # Number of matrices each GPU is responsible for (rounded up)
        chunk_size = (len(params) + world_size - 1) // world_size
        # Total after padding to a multiple of world_size
        padded_num_params = chunk_size * world_size
        # Reference parameter
        p = params[0]
        # Its shape, device and dtype
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        # Stack all gradients
        grad_stack = torch.stack([p.grad for p in params])
        # Allocate padded buffer
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        # Copy the real gradients in
        stacked_grads[:len(params)].copy_(grad_stack)
        # If padding is needed...
        if len(params) < padded_num_params:
            # ...fill padding with zeros
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        # Buffer for this GPU's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        # Average across GPUs and scatter chunks
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        # Return bookkeeping
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    # Phase 2 for AdamW: wait for gradients, apply the update
    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        # Bookkeeping from phase 1
        param_infos = info['param_infos']
        # For each parameter
        for p in group['params']:
            # Its info
            pinfo = param_infos[p]
            # If there was a pending communication...
            if pinfo['future'] is not None:
                # ...wait for it to finish
                pinfo['future'].wait()
            # The averaged gradient (slice)
            grad_slice = pinfo['grad_slice']
            # Optimizer state (m, v, step) for this parameter
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            # Small: update whole tensor
            if pinfo['is_small']:
                # Whole tensor
                p_slice = p
            # Large: update only our slice
            else:
                # Rows per GPU
                rank_size = p.shape[0] // world_size
                # Our rows
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            # First step: create the state
            if not state:
                # Step counter
                state['step'] = 0
                # m = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                # v = 0
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            # Increment step
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            # Put step in holder
            self._adamw_step_t.fill_(state['step'])
            # Put LR in holder
            self._adamw_lr_t.fill_(group['lr'])
            # Put beta1 in holder
            self._adamw_beta1_t.fill_(group['betas'][0])
            # Put beta2 in holder
            self._adamw_beta2_t.fill_(group['betas'][1])
            # Put eps in holder
            self._adamw_eps_t.fill_(group['eps'])
            # Put weight decay in holder
            self._adamw_wd_t.fill_(group['weight_decay'])
            # Run the fused AdamW update
            adamw_step_fused(
                # Tensors
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                # Scalars
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                # More scalars
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            # Close call
            )

            # Large params need all_gather
            # If we only updated a slice...
            if not pinfo['is_small']:
                # ...send our updated slice to every GPU and collect theirs
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                # Track the pending gather
                gather_list.append(dict(future=future, params=None))

    # Phase 2 for Muon: wait for gradients, orthogonalize, update
    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        # If communication is pending...
        if info['future'] is not None:
            # ...wait for it
            info['future'].wait()
        # Parameters in this group
        params = group['params']
        # Matrices per GPU
        chunk_size = info['chunk_size']
        # Our chunk of averaged gradients
        grad_chunk = info['grad_chunk']
        # Reference param
        p = params[0]
        # Shape, device, dtype
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        # Index of our first matrix
        start_idx = rank * chunk_size
        # How many real (non-padding) matrices we own
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        # State is stored on the first param of the group
        state = self.state[p]
        # Create momentum buffer if missing
        if "momentum_buffer" not in state:
            # Zeros, one matrix per owned slot
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        # Create second-moment buffer if missing
        if "second_momentum_buffer" not in state:
            # One value per row (tall) or per column (wide): much cheaper than a full matrix
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            # Zeros
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        # Average over columns for tall, rows for wide
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Placeholder
        stacked_owned = None
        # If we own any matrices
        if num_owned > 0:
            # List of the matrices we own
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            # Stack them into one tensor
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            # Momentum
            self._muon_momentum_t.fill_(group["momentum"])
            # beta2
            self._muon_beta2_t.fill_(group["beta2"])
            # LR scaled up by sqrt(rows/cols) for tall matrices (keeps update size consistent across shapes)
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            # Weight decay
            self._muon_wd_t.fill_(group["weight_decay"])
            # Run the fused Muon update
            muon_step_fused(
                # Gradients and params
                grad_chunk[:num_owned], stacked_owned,
                # State buffers
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                # Scalars
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                # Iterations and reduction dim
                group["ns_steps"], red_dim,
            # Close call
            )

        # Single-GPU case
        if info['stacked_grads'] is None:
            # Single rank: no gather needed, the updated stack maps directly onto the params
            # Queue copying the stack back into the individual params
            gather_list.append(dict(future=None, stacked_params=stacked_owned, params=params))
            # Done
            return

        # Build the input buffer for all_gather
        # Buffer of our updated matrices
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        # If we own any
        if num_owned > 0:
            # Copy them in
            updated_params[:num_owned].copy_(stacked_owned)
        # If there is padding
        if num_owned < chunk_size:
            # Zero it
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        # Reuse memory
        stacked_params = info["stacked_grads"]
        # Gather every GPU's updated matrices
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        # Track pending gather
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    # Phase 3: wait and copy updated weights back
    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        # For each pending item
        for info in gather_list:
            # If communication pending
            if info["future"] is not None:
                # Wait
                info["future"].wait()
            # Muon entries need copying back
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                # Unstack and copy each matrix into its parameter
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    # The optimizer step doesn't need gradient tracking
    @torch.no_grad()
    # Called once per training iteration after loss.backward()
    def step(self):
        # On a single rank (no multi-rank process group), all communication is skipped
        # If multi-GPU training is running
        if dist.is_available() and dist.is_initialized():
            # This GPU's index
            rank = dist.get_rank()
            # Total GPUs
            world_size = dist.get_world_size()
        # Single GPU/CPU
        else:
            # Index 0
            rank = 0
            # One worker
            world_size = 1

        # Phase 1: launch all async reduce ops
        # Bookkeeping per group
        reduce_infos: list[dict] = []
        # For each group
        for group in self.param_groups:
            # AdamW group
            if group['kind'] == 'adamw':
                # Start its reduce
                reduce_infos.append(self._reduce_adamw(group, world_size))
            # Muon group
            elif group['kind'] == 'muon':
                # Start its reduce
                reduce_infos.append(self._reduce_muon(group, world_size))
            # Unknown
            else:
                # Error out
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        # Pending gathers
        gather_list: list[dict] = []
        # For each group with its info
        for group, info in zip(self.param_groups, reduce_infos):
            # AdamW
            if group['kind'] == 'adamw':
                # Compute update
                self._compute_adamw(group, info, gather_list, rank, world_size)
            # Muon
            elif group['kind'] == 'muon':
                # Compute update
                self._compute_muon(group, info, gather_list, rank)
            # Unknown
            else:
                # Error out
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        # Finish
        self._finish_gathers(gather_list)
