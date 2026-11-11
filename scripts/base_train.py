"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""
# =====================================================================================
# STUDY NOTES (added): this is THE pretraining script. The core loop at the bottom is just:
#     loss = model(x, y)      # forward pass: lots of matrix multiplies -> one number (how wrong we are)
#     loss.backward()         # backward pass: chain rule -> gradient dLoss/dW for every weight (~2x forward cost)
#     optimizer.step()        # move every weight a little bit downhill
#     repeat millions of times on billions of tokens.
# Total compute ~= 6 * N_params * N_tokens FLOPs. Everything else here is bookkeeping, scheduling and logging.
# =====================================================================================

# Import os for environment variables and file paths
import os
# Tell PyTorch's GPU memory allocator to use growable segments (less fragmentation / fewer out-of-memory errors)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Import the garbage collector (manually controlled later for speed)
import gc
# Import json for pretty-printing configs
import json
# Import time for measuring step durations
import time
# Import math for log2, sqrt, cos
import math
# Import argparse for command-line flags
import argparse
# Import asdict to turn a dataclass into a dictionary
from dataclasses import asdict
# Import contextmanager to write "with ..." helpers
from contextlib import contextmanager

# Import Weights & Biases (online experiment logging/plots)
import wandb
# Import PyTorch
import torch
# Import distributed communication between GPUs
import torch.distributed as dist

# Import the model class, its config, and our Linear layer
from nanochat.gpt import GPT, GPTConfig, Linear
# Import data loaders that stream tokenized text and pack it into (B, T) batches
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
# Import setup/teardown helpers, logging helpers, device detection and the GPU's peak FLOPS table
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
# Import the tokenizer and per-token byte lengths (for the bits-per-byte metric)
from nanochat.tokenizer import get_tokenizer, get_token_bytes
# Import checkpoint save/load
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
# Import the validation metric (bits per byte)
from nanochat.loss_eval import evaluate_bpb
# Import the inference engine (used to sample text during training)
from nanochat.engine import Engine
# Import whether Flash Attention 3 is available
from nanochat.flash_attention import HAS_FA3
# Import the CORE benchmark evaluation
from scripts.base_eval import evaluate_core
# Print the nanochat ASCII banner
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
# Create the command-line argument parser
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
# Name of the wandb run ("dummy" = no logging)
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
# Which hardware to use
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
# Use 8-bit matmuls (faster on H100)
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
# FP8 scaling method
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
# Number of layers: THE single knob that sets model size in nanochat (width is derived from it)
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
# Width = depth * 64. So params ~ 12 * depth * (64*depth)^2 ~ depth^3: doubling depth => ~8x params
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
# Size of each attention head
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
# Context length T
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# Attention window pattern across layers
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
# Fixed number of steps (overrides the others)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
# Train until this many total FLOPs are spent
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
# Train on (ratio x params) tokens. Chinchilla found ~20 tokens/param is compute-optimal
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
# Sequences per GPU per forward pass (limited by GPU memory)
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
# Tokens per optimizer step (across all GPUs and accumulation steps)
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
# LR for token embeddings
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
# LR for the output head
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
# Weight decay for matrices
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
# LR for the big matrices (Muon)
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
# LR for scalars
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
# Steps to ramp LR up from ~0 (avoids huge unstable updates at the start)
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
# Fraction of training spent lowering the LR at the end (helps settle into a minimum)
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
# LR at the very end as a fraction of the peak
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
# Resume from a saved step
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
# How often to measure validation loss
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
# How many tokens to measure it on
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
# How often to run the CORE benchmark
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
# Examples per CORE task
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
# How often to print sample generations
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
# How often to save checkpoints
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
# Custom checkpoint folder name
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# Parse the command line
args = parser.parse_args()
# Copy of all settings for logging
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

# Pick the device: auto-detect unless specified
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# Initialize (possibly multi-GPU) runtime: are we distributed, which GPU am I, how many GPUs, torch device
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# Only GPU 0 logs and saves
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
# GPU work is asynchronous; synchronize() waits for it to finish (needed for accurate timing)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# Function to read peak GPU memory used
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
# On NVIDIA GPUs...
if device_type == "cuda":
    # ...read the GPU model name
    gpu_device_name = torch.cuda.get_device_name(0)
    # ...look up its theoretical peak bf16 FLOPs/second (e.g. H100 ~ 989e12)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    # ...print it
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
# On CPU/Mac...
else:
    # ...MFU (model FLOPs utilization) isn't meaningful
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
# Report the math precision being used
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
# Use a do-nothing logger unless a run name was given and we are GPU 0
use_dummy_wandb = args.run == "dummy" or not master_process
# Create the logger
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
# Import the final decision on FA3
from nanochat.flash_attention import USE_FA3
# Store it
using_fa3 = USE_FA3
# If FA3 is in use...
if using_fa3:
    # ...say so
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
# Otherwise warn
else:
    # Banner line
    print0("!" * 80)
    # FA3 exists but wrong dtype
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        # Explain
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    # FA3 not installed/supported
    else:
        # Explain
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    # Slower training warning
    print0("WARNING: Training will be less efficient without FA3")
    # Sliding windows need explicit masks in SDPA (slow)
    if args.window_pattern != "L":
        # Warn
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        # Recommend
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    # Banner line
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
# Load the trained BPE tokenizer (text <-> token ids)
tokenizer = get_tokenizer()
# Number of UTF-8 bytes each token represents (to convert loss into bits per byte)
token_bytes = get_token_bytes(device=device)
# Vocabulary size V
vocab_size = tokenizer.get_vocab_size()
# Print it
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

# Build a model "skeleton" (shapes only) for a given depth
def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    # Raw width = depth * 64
    base_dim = depth * args.aspect_ratio
    # Round width up to a multiple of head_dim (128)
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    # Number of heads = width / head_dim
    num_heads = model_dim // args.head_dim
    # Build the config
    config = GPTConfig(
        # Context length and vocab
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        # Layers, heads (no GQA here), width
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        # Window pattern
        window_pattern=args.window_pattern,
    # Close
    )
    # On the meta device no memory is allocated
    with torch.device("meta"):
        # Build the model skeleton
        model_meta = GPT(config)
    # Return it
    return model_meta

# Build the model, move to device, init the weights
# Step 1: skeleton
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
# The config object
model_config = model.config
# Config as a dict
model_config_kwargs = asdict(model_config)
# Print config
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
# Step 2: allocate real (uninitialized) memory on the GPU
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
# Step 3: fill weights with their starting random values
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
# Root folder for nanochat data (~/.cache/nanochat by default)
base_dir = get_base_dir()
# Checkpoint subfolder name, e.g. "d20"
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
# Full checkpoint path
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
# Are we resuming?
resuming = args.resume_from_step != -1
# If so...
if resuming:
    # ...announce it
    print0(f"Resuming optimization from step {args.resume_from_step}")
    # ...load weights, optimizer state and metadata
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    # ...put the weights into the model
    model.load_state_dict(model_data, strict=True, assign=True)
    # ...free the temporary copy
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
# If FP8 requested...
if args.fp8:
    # ...it needs an NVIDIA GPU
    if device_type != "cuda":
        # Warn and ignore
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    # Otherwise set it up
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        # Import our FP8 helpers
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        # Import nn for type checks
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        # Decide which Linear layers to convert
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            # Only Linear layers
            if not isinstance(mod, nn.Linear):
                # Skip others
                return False
            # FP8 tensor cores need dimensions divisible by 16
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                # Skip
                return False
            # Tiny matmuls aren't worth it (quantization overhead > savings)
            if min(mod.in_features, mod.out_features) < 128:
                # Skip
                return False
            # Convert this one
            return True

        # Build config
        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        # Count Linear layers before
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        # Convert eligible layers
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        # Count converted layers
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        # Count skipped
        num_skipped = num_linear - num_fp8
        # Report
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
# Make the function usable with "with disable_fp8(model):"
@contextmanager
# Temporarily swap FP8 layers back to bf16 layers
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    # Import nn
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    # List of where each FP8 layer lives
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    # Walk every submodule with its dotted name
    for name, module in model.named_modules():
        # If it's an FP8 layer...
        if 'Float8' in type(module).__name__:
            # Nested module
            if '.' in name:
                # Split "a.b.c" into parent "a.b" and attribute "c"
                parent_name, attr_name = name.rsplit('.', 1)
                # Get the parent module
                parent = model.get_submodule(parent_name)
            # Top-level module
            else:
                # Parent is the model itself
                parent = model
                # Attribute name
                attr_name = name
            # Remember location
            fp8_locations.append((parent, attr_name, module))

    # No FP8 layers at all
    if not fp8_locations:
        # Run the with-block body unchanged
        yield  # No FP8 modules, nothing to do
        # Done
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    # For each FP8 layer
    for parent, attr_name, fp8_module in fp8_locations:
        # Create a regular Linear shell
        linear = Linear(
            # Same input size
            fp8_module.in_features,
            # Same output size
            fp8_module.out_features,
            # Same bias setting
            bias=fp8_module.bias is not None,
            # No memory allocated
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            # Same dtype
            dtype=fp8_module.weight.dtype,
        # Close
        )
        # Share the same weight tensor
        linear.weight = fp8_module.weight  # share, don't copy
        # If there's a bias...
        if fp8_module.bias is not None:
            # ...share it too
            linear.bias = fp8_module.bias
        # Put the bf16 layer in place of the FP8 one
        setattr(parent, attr_name, linear)

    # Run the with-block body, then always restore
    try:
        # Body runs here
        yield
    # Always executed
    finally:
        # Restore Float8Linear modules
        # For each swapped layer
        for parent, attr_name, fp8_module in fp8_locations:
            # Put the FP8 layer back
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

# Keep a reference to the plain model (for saving and variable-shape inference)
orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
# torch.compile fuses many small GPU ops into fewer big kernels (less memory traffic => faster)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
# Parameter counts by group
param_counts = model.num_scaling_params()
# Header
print0(f"Parameter counts:")
# For each group...
for key, value in param_counts.items():
    # ...print its count
    print0(f"{key:24s}: {value:,}")
# Total parameters
num_params = param_counts['total']
# FLOPs per training token (~6 * matmul params + attention)
num_flops_per_token = model.estimate_flops()
# Print it
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
# Which parameter count to use for scaling-law calculations
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    # Get counts
    params_counts = m.num_scaling_params()
    # Use block matrices + output head
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    # Return
    return scaling_params
# Scaling params for our model
num_scaling_params = get_scaling_params(model)
# Tokens to train on = ratio x params. Bigger model => more tokens AND more FLOPs/token => compute grows ~ params^2
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
# Build a depth-12 reference skeleton
d12_ref = build_model_meta(12) # creates the model on meta device
# Reference token horizon
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
# Reference batch size (tokens)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
# User may override
total_batch_size = args.total_batch_size # user-provided override is possible
# Auto-compute if not given
if total_batch_size == -1:
    # How many times more data than the reference
    batch_size_ratio = target_tokens / D_REF
    # Power-law prediction of the best batch size
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    # Round to nearest power of 2
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    # Report
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
# Default: no change
batch_lr_scale = 1.0
# Batch size relative to the reference
batch_ratio = total_batch_size / B_REF # B/B_ref
# If different from the reference...
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    # Bigger batch => gradient averaged over more tokens => less noise => can take larger steps
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    # Report
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
# Scaled weight decay
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
# If it changed...
if weight_decay_scaled != args.weight_decay:
    # ...report
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Build the optimizer with batch-scaled learning rates
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    # Output head LR
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    # Embedding LR
    embedding_lr=args.embedding_lr * batch_lr_scale,
    # Scalar LR
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    # Matrix LR
    matrix_lr=args.matrix_lr * batch_lr_scale,
    # Weight decay
    weight_decay=weight_decay_scaled,
# Close
)

# If resuming...
if resuming:
    # ...restore optimizer state (momentum buffers etc.)
    optimizer.load_state_dict(optimizer_data)
    # ...free memory
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
# fp16 can't represent tiny gradients (they underflow to 0), so multiply the loss by a big factor first
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
# If used...
if scaler is not None:
    # ...report
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
# Where to resume reading data from (or None)
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
# Infinite generator of (x, y) training batches. y is x shifted by one token: the "next token" targets
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
# Function that makes a fresh validation loader each time
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
# Fetch the first batch
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
# At least one horizon must be set
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
# Explicit step count
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    # Use it
    num_iterations = args.num_iterations
    # Report
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
# FLOP budget
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    # steps = total FLOPs / (FLOPs per token * tokens per step)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    # Report
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
# Data:param ratio (default)
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    # steps = target tokens / tokens per step
    num_iterations = target_tokens // total_batch_size
    # Report
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
# Nothing set
else:
    # Error
    raise ValueError("No training horizon specified")
# Actual total tokens
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
# Report
print0(f"Total number of training tokens: {total_tokens:,}")
# Report tokens per parameter
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
# Report total compute: this number divided by (GPU FLOPs/sec x utilization) = training time
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Multiplier applied to every group's LR at step `it`
def get_lr_multiplier(it):
    # Warmup length
    warmup_iters = args.warmup_steps
    # Warmdown length
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    # During warmup...
    if it < warmup_iters:
        # ...ramp linearly from ~0 to 1
        return (it + 1) / warmup_iters
    # Middle phase...
    elif it <= num_iterations - warmdown_iters:
        # ...full LR
        return 1.0
    # Warmdown phase...
    else:
        # ...fraction of warmdown remaining (1 -> 0)
        progress = (num_iterations - it) / warmdown_iters
        # ...linearly decrease from 1 to final_lr_frac (smaller steps let the weights settle into a valley of the loss)
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
# Muon momentum at step `it`
def get_muon_momentum(it):
    # Warmdown length
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    # Warmdown start step
    warmdown_start = num_iterations - warmdown_iters
    # First 400 steps...
    if it < 400:
        # ...fraction through
        frac = it / 400
        # ...ramp momentum 0.85 -> 0.97
        return (1 - frac) * 0.85 + frac * 0.97
    # During warmdown...
    elif it >= warmdown_start:
        # ...fraction through warmdown
        progress = (it - warmdown_start) / warmdown_iters
        # ...decrease 0.97 -> 0.90
        return 0.97 * (1 - progress) + 0.90 * progress
    # Middle phase
    else:
        # Constant
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
# Weight decay at step `it`
def get_weight_decay(it):
    # Half-cosine from full value down to 0
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# Loop state (variables updated by the training loop)
# Fresh run
if not resuming:
    # Step counter
    step = 0
    # Latest validation bits-per-byte
    val_bpb = None # will be set if eval_every > 0
    # Best validation bpb seen
    min_val_bpb = float("inf")
    # Smoothed training loss for logging
    smooth_train_loss = 0 # EMA of training loss
    # Wall-clock training time
    total_training_time = 0 # total wall-clock time of training
# Resumed run
else:
    # Restore step
    step = meta_data["step"]
    # Restore loop state dict
    loop_state = meta_data["loop_state"]
    # Restore val bpb
    val_bpb = meta_data["val_bpb"]
    # Restore best val bpb
    min_val_bpb = loop_state["min_val_bpb"]
    # Restore smoothed loss
    smooth_train_loss = loop_state["smooth_train_loss"]
    # Restore training time
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
# Tokens one GPU processes in one forward/backward
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
# Tokens all GPUs process together in one forward/backward
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
# Total batch must be an exact multiple
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
# If the batch doesn't fit in memory at once, run several forward/backward passes and SUM gradients before stepping
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
# Report
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
# Report
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
# Report
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
# Main training loop
while True:
    # Is this the final iteration (only eval/save, no training)?
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    # Compute spent so far
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    # Time to evaluate?
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        # Switch to evaluation mode
        model.eval()
        # Fresh validation data
        val_loader = build_val_loader()
        # Number of eval batches
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        # Evaluate in bf16
        with disable_fp8(model):
            # Bits per byte on held-out text (lower = better compression = better model)
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        # Report
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        # New best?
        if val_bpb < min_val_bpb:
            # Record it
            min_val_bpb = val_bpb
        # Log to wandb
        wandb_run.log({
            # Step
            "step": step,
            # Compute
            "total_training_flops": flops_so_far,
            # Time
            "total_training_time": total_training_time,
            # Metric
            "val/bpb": val_bpb,
        # Close
        })
        # Back to training mode
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    # Benchmark results
    results = {}
    # Time for CORE eval?
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        # Eval mode
        model.eval()
        # bf16
        with disable_fp8(orig_model):
            # Run the benchmark suite
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        # Report
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        # Log
        wandb_run.log({
            # Step
            "step": step,
            # Compute
            "total_training_flops": flops_so_far,
            # Score
            "core_metric": results["core_metric"],
            # Per-task scores
            "centered_results": results["centered_results"],
        # Close
        })
        # Training mode
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    # Time to print samples?
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        # Eval mode
        model.eval()
        # Test prompts
        prompts = [
            # Prompt
            "The capital of France is",
            # Prompt
            "The chemical symbol of gold is",
            # Prompt
            "If yesterday was Friday, then tomorrow will be",
            # Prompt
            "The opposite of hot is",
            # Prompt
            "The planets of the solar system are:",
            # Prompt
            "My favorite color is",
            # Prompt
            "If 5*x + 3 = 13, then x is",
        # Close
        ]
        # Inference engine
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        # For each prompt
        for prompt in prompts:
            # Tokenize with a beginning-of-sequence token
            tokens = tokenizer(prompt, prepend="<|bos|>")
            # bf16
            with disable_fp8(orig_model):
                # Greedy-generate 16 tokens
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            # Print the text
            print0(tokenizer.decode(sample[0]))
        # Training mode
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    # Time to save?
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        # Save everything
        save_checkpoint(
            # Folder
            checkpoint_dir,
            # Step
            step,
            # Weights
            orig_model.state_dict(), # model parameters
            # Optimizer state
            optimizer.state_dict(), # optimizer state
            # Metadata
            { # metadata saved as json
                # Step
                "step": step,
                # Latest val metric
                "val_bpb": val_bpb, # loss at last step
                # Model shape
                "model_config": model_config_kwargs,
                # CLI args
                "user_config": user_config, # inputs to the training script
                # Batch
                "device_batch_size": args.device_batch_size,
                # Context
                "max_seq_len": args.max_seq_len,
                # Total batch
                "total_batch_size": total_batch_size,
                # Data position
                "dataloader_state_dict": dataloader_state_dict,
                # Loop state
                "loop_state": { # all loop state (other than step) so that we can resume training
                    # Best val
                    "min_val_bpb": min_val_bpb,
                    # Smoothed loss
                    "smooth_train_loss": smooth_train_loss,
                    # Time
                    "total_training_time": total_training_time,
                # Close
                },
            # Close
            },
            # Which GPU
            rank=ddp_rank,
        # Close
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    # Final iteration...
    if last_step:
        # ...exit the loop
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    # Wait for GPU to be idle before timing
    synchronize()
    # Start timer
    t0 = time.time()
    # Accumulate gradients over several micro-batches
    for micro_step in range(grad_accum_steps):
        # FORWARD: x (B,T) token ids -> scalar cross-entropy loss vs targets y
        loss = model(x, y)
        # Copy for logging (detached from the autograd graph)
        train_loss = loss.detach() # for logging
        # Divide so the SUM of gradients over micro-steps equals the gradient of the MEAN loss
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        # fp16 path
        if scaler is not None:
            # Scale up loss, then BACKWARD
            scaler.scale(loss).backward()
        # bf16 path
        else:
            # BACKWARD: chain rule through every layer, adds dLoss/dW into each weight's .grad
            loss.backward()
        # Load the next batch (CPU) while the GPU is still busy
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    # LR multiplier for this step
    lrm = get_lr_multiplier(step)
    # Muon momentum for this step
    muon_momentum = get_muon_momentum(step)
    # Weight decay for this step
    muon_weight_decay = get_weight_decay(step)
    # For every parameter group...
    for group in optimizer.param_groups:
        # ...set its LR
        group["lr"] = group["initial_lr"] * lrm
        # Muon groups also get...
        if group['kind'] == 'muon':
            # ...momentum
            group["momentum"] = muon_momentum
            # ...and weight decay
            group["weight_decay"] = muon_weight_decay
    # fp16 path
    if scaler is not None:
        # Divide gradients by the scale factor
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        # Multi-GPU...
        if is_ddp_initialized():
            # For each device's inf flag...
            for v in scaler._found_inf_per_device(optimizer).values():
                # ...share it (if any GPU overflowed, all skip)
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        # Step (skipped if gradients overflowed)
        scaler.step(optimizer)
        # Adjust the scale factor
        scaler.update()
    # bf16 path
    else:
        # UPDATE: W <- W - lr * update(gradient) for every weight
        optimizer.step()
    # Clear gradients for the next step (set to None frees memory)
    model.zero_grad(set_to_none=True)
    # Move loss to CPU as a Python float (forces GPU sync)
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    # Wait for GPU
    synchronize()
    # Stop timer
    t1 = time.time()
    # Step duration in seconds
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    # Smoothing factor
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    # Exponential moving average of the loss
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    # Correct the startup bias (same idea as Adam's bias correction)
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    # Percent complete
    pct_done = 100 * step / num_iterations
    # Throughput in tokens/second
    tok_per_sec = int(total_batch_size / dt)
    # Achieved FLOPs/second
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    # MFU = achieved / theoretical peak. ~40-50% is good; the rest is lost to memory traffic, communication, etc.
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    # Skip the first steps (compilation warm-up is slow)
    if step > 10:
        # Accumulate time
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    # Steps counted
    steps_done = step - 10
    # If any...
    if steps_done > 0:
        # Average step time
        avg_time_per_step = total_training_time / steps_done
        # Steps left
        remaining_steps = num_iterations - step
        # Estimated seconds left
        eta_seconds = remaining_steps * avg_time_per_step
        # Format
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    # Not yet
    else:
        # Empty
        eta_str = ""
    # Position in the dataset
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    # Print progress line
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    # Every 100 steps...
    if step % 100 == 0:
        # ...build log record
        log_data = {
            # Step
            "step": step,
            # Compute
            "total_training_flops": flops_so_far,
            # Time
            "total_training_time": total_training_time,
            # Loss
            "train/loss": debiased_smooth_loss,
            # LR multiplier
            "train/lrm": lrm,
            # Step time
            "train/dt": dt,
            # Throughput
            "train/tok_per_sec": tok_per_sec,
            # Utilization
            "train/mfu": mfu,
            # Data position
            "train/epoch": epoch,
        # Close
        }
        # Send
        wandb_run.log(log_data)

    # state update
    # Was this the first step of this run?
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    # Next step
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    # After the first step...
    if first_step_of_run:
        # ...clean up setup garbage
        gc.collect() # manually collect a lot of garbage from setup
        # ...mark surviving objects as permanent
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        # ...turn off automatic GC
        gc.disable() # nuclear intervention here: disable GC entirely except:
    # Occasionally...
    elif step % 5000 == 0: # every 5000 steps...
        # ...collect manually
        gc.collect() # manually collect, just to be safe for very, very long runs

# print a few more stats
# Peak GPU memory
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
# Total time
print0(f"Total training time: {total_training_time/60:.2f}m")
# If evaluated...
if val_bpb is not None:
    # ...best validation score
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
# Close wandb
wandb_run.finish() # wandb run finish
# Shut down distributed processes
compute_cleanup()
