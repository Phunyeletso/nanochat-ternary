"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""
# STUDY NOTE (added): SFT uses EXACTLY the same math as pretraining (forward -> cross-entropy -> backward -> optimizer step).
# Only the data changes: chat conversations instead of web text, and the loss is only computed on the
# assistant's tokens (targets = -1 elsewhere are ignored). It is cheap: a few % of pretraining compute.

# Import garbage collector
import gc
# Import argparse
import argparse
# Import os
import os
# Reduce GPU memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Import time
import time
# Import wandb logging
import wandb
# Import PyTorch
import torch
# Import setup/logging helpers
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
# Import bytes-per-token table
from nanochat.tokenizer import get_token_bytes
# Import checkpoint helpers
from nanochat.checkpoint_manager import save_checkpoint, load_model, load_optimizer_state
# Import bits-per-byte evaluation
from nanochat.loss_eval import evaluate_bpb
# Import distributed communication
import torch.distributed as dist
# Import FA3 availability flag
from nanochat.flash_attention import HAS_FA3
# Import inference engine
from nanochat.engine import Engine
# Import chat evaluation
from scripts.chat_eval import run_chat_eval

# Import a helper that mixes several datasets
from tasks.common import TaskMixture
# Import grade-school math dataset
from tasks.gsm8k import GSM8K
# Import multiple-choice knowledge dataset
from tasks.mmlu import MMLU
# Import general conversation dataset
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# CLI arguments
# Parser
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
# wandb run name
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
# Device
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
# Which pretrained model folder
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
# Which step
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Reuse optimizer momentum from pretraining?
parser.add_argument("--load-optimizer", type=int, default=1, help="warm-start optimizer from pretrained checkpoint (0=no, 1=yes)")
# Training horizon
# Steps (-1 = one pass over the data)
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes (default: inherit from pretrained checkpoint)
# Context length
parser.add_argument("--max-seq-len", type=int, default=None, help="max context length (default: inherit from pretrain)")
# Per-GPU batch
parser.add_argument("--device-batch-size", type=int, default=None, help="per-device batch size (default: inherit from pretrain)")
# Total tokens per step
parser.add_argument("--total-batch-size", type=int, default=None, help="total batch size in tokens (default: inherit from pretrain)")
# Optimization (default: inherit from pretrained checkpoint)
# Embedding LR
parser.add_argument("--embedding-lr", type=float, default=None, help="learning rate for embedding parameters (Adam) (default: inherit from pretrain)")
# Output head LR
parser.add_argument("--unembedding-lr", type=float, default=None, help="learning rate for unembedding parameters (Adam) (default: inherit from pretrain)")
# Matrix LR
parser.add_argument("--matrix-lr", type=float, default=None, help="learning rate for matrix parameters (Muon) (default: inherit from pretrain)")
# Start at 80% of pretraining LR
parser.add_argument("--init-lr-frac", type=float, default=0.8, help="initial LR as fraction of base LR")
# Warmup fraction
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
# Warmdown fraction
parser.add_argument("--warmdown-ratio", type=float, default=0.5, help="ratio of iterations for LR warmdown")
# Final LR fraction
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
# Evaluation
# Val frequency
parser.add_argument("--eval-every", type=int, default=200, help="evaluate val bpb every N steps (-1 = disable)")
# Val tokens
parser.add_argument("--eval-tokens", type=int, default=40*524288, help="number of tokens to evaluate val loss on")
# ChatCORE frequency
parser.add_argument("--chatcore-every", type=int, default=200, help="evaluate ChatCORE metric every N steps (-1 = disable)")
# Limit for multiple-choice tasks
parser.add_argument("--chatcore-max-cat", type=int, default=-1, help="max problems per categorical task for ChatCORE")
# Limit for generative tasks
parser.add_argument("--chatcore-max-sample", type=int, default=24, help="max problems per generative task for ChatCORE")
# Data mixture
# How many times to repeat MMLU
parser.add_argument("--mmlu-epochs", type=int, default=3, help="number of epochs of MMLU in training mixture (teaches Multiple Choice)")
# How many times to repeat GSM8K
parser.add_argument("--gsm8k-epochs", type=int, default=4, help="number of epochs of GSM8K in training mixture (teaches Math and Tool Use)")
# Parse
args = parser.parse_args()
# Copy for logging
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
# Device type
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# Init runtime
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# Is this GPU 0?
master_process = ddp_rank == 0
# Report dtype
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
# GPU sync function
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
# Peak memory function
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
# On GPU...
if device_type == "cuda":
    # ...GPU name
    gpu_device_name = torch.cuda.get_device_name(0)
    # ...peak FLOPs
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    # ...report
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
# Otherwise
else:
    # MFU not meaningful
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS

# wandb logging init
# Use dummy logger?
use_dummy_wandb = args.run == "dummy" or not master_process
# Create logger
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config)

# Flash Attention status
# Without FA3...
if not HAS_FA3:
    # ...warn
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient.")

# Load the model and tokenizer
# Start from the PRETRAINED weights (fine-tuning, not training from scratch)
model, tokenizer, meta = load_model("base", device, phase="train", model_tag=args.model_tag, step=args.model_step)

# Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
# Pretraining CLI settings
pretrain_user_config = meta.get("user_config", {})
# For each inheritable setting (name, default, where to look)
for name, fallback, source in [
    # Context length
    ("max_seq_len",       2048,  meta),
    # Per-GPU batch
    ("device_batch_size", 32,    meta),
    # Total batch
    ("total_batch_size",  524288, meta),
    # Embedding LR
    ("embedding_lr",      0.3,   pretrain_user_config),
    # Output head LR
    ("unembedding_lr",    0.004, pretrain_user_config),
    # Matrix LR
    ("matrix_lr",         0.02,  pretrain_user_config),
# Close list
]:
    # Value given on the command line (or None)
    arg_val = getattr(args, name)
    # Value used in pretraining (or None)
    pretrain_val = source.get(name)
    # Not given on CLI
    if arg_val is None:
        # Use pretraining value, else default
        resolved = pretrain_val if pretrain_val is not None else fallback
        # Store it
        setattr(args, name, resolved)
        # Report
        print0(f"Inherited {name}={resolved} from pretrained checkpoint")
    # Given and different
    elif pretrain_val is not None and arg_val != pretrain_val:
        # Report override
        print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
    # Given and same (or no pretraining value)
    else:
        # Report
        print0(f"Using {name}={arg_val}")

# Keep uncompiled model
orig_model = model
# Compile for speed
model = torch.compile(model, dynamic=False)
# Number of layers
depth = model.config.n_layer
# FLOPs per token
num_flops_per_token = model.estimate_flops()
# Tokens per GPU per forward/backward
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
# Tokens across all GPUs
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
# Must divide evenly
assert args.total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({args.total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
# Gradient accumulation steps
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
# Report
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
# Report
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
# Report
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
# Bytes per token table
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
# Build optimizer (no weight decay)
optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=0.0)

# Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
# Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
# pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
# restore our fresh SFT LRs after loading.
# Base dir
base_dir = get_base_dir()
# If warm-starting the optimizer
if args.load_optimizer:
    # Load this GPU's shard
    optimizer_data = load_optimizer_state("base", device, rank=ddp_rank, model_tag=args.model_tag, step=args.model_step)
    # Found it
    if optimizer_data is not None:
        # Remember our fresh LRs
        base_lrs = [group["lr"] for group in optimizer.param_groups]
        # Load momentum buffers etc.
        optimizer.load_state_dict(optimizer_data)
        # Free memory
        del optimizer_data
        # For each group
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            # Restore our LR
            group["lr"] = base_lr
        # Report
        print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
    # Not found
    else:
        # Warn
        print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

# GradScaler for fp16 training (bf16/fp32 don't need it)
# fp16 loss scaler
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
# If used
if scaler is not None:
    # Report
    print0("GradScaler enabled for fp16 training")

# Override the initial learning rate as a fraction of the base learning rate
# For each group
for group in optimizer.param_groups:
    # Scale LR down
    group["lr"] = group["lr"] * args.init_lr_frac
    # Remember as initial LR
    group["initial_lr"] = group["lr"]

# SFT data mixture and DataLoader
# Training datasets
train_tasks = [
    # General chat
    SmolTalk(split="train"), # 460K rows of general conversations
    # Multiple choice, repeated
    *[MMLU(subset="all", split="auxiliary_train") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
    # Math with tool use, repeated
    *[GSM8K(subset="main", split="train") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
# Close
]
# Shuffled mixture
train_dataset = TaskMixture(train_tasks)
# Report
print0(f"Training mixture: {len(train_dataset):,} rows (MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs})")
# Validation mixture
val_dataset = TaskMixture([
    # Chat test set
    SmolTalk(split="test"), # 24K rows in test set
    # MMLU test subset
    MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
    # GSM8K test subset
    GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
# Close
]) # total: 24K + 5.2K + 0.42K ~= 29.6K rows
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
# Set when data runs out
last_step = False # we will toggle this to True when we reach the end of the training dataset
# Fraction of the epoch done
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
# Epoch counter
current_epoch = 1 # track epoch for logging
# Generator packing conversations into (B, T) batches
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    # Modify the globals
    global last_step, approx_progress, current_epoch
    # Validate
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    # Pick dataset
    dataset = train_dataset if split == "train" else val_dataset
    # Number of conversations
    dataset_size = len(dataset)
    # Non-empty
    assert dataset_size > 0
    # T+1 tokens per row
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    # BOS id
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (token_ids, loss_mask) tuples
    # Pool
    conv_buffer = []
    # Next conversation index to fetch (each GPU starts at its rank)
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    # Conversations actually used
    consumed = ddp_rank  # Track actual consumption separately from buffering
    # Epoch
    epoch = 1
    # Batch counter
    it = 0  # iteration counter

    # Top up the pool
    def refill_buffer():
        # Modify outer variables
        nonlocal cursor, epoch
        # Until full
        while len(conv_buffer) < buffer_size:
            # Fetch conversation
            conversation = dataset[cursor]
            # Tokenize with training mask
            ids, mask = tokenizer.render_conversation(conversation)
            # Add to pool
            conv_buffer.append((ids, mask))
            # Stride by number of GPUs
            cursor += ddp_world_size
            # Wrapped around?
            if cursor >= dataset_size:
                # Wrap
                cursor = cursor % dataset_size
                # Next epoch
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching

    # Forever
    while True:
        # Token rows
        rows = []
        # Mask rows
        mask_rows = []
        # Real content length per row
        row_lengths = []  # Track actual content length (excluding padding) for each row
        # For each of B rows
        for _ in range(args.device_batch_size):
            # Tokens
            row = []
            # Mask
            mask_row = []
            # Did we pad?
            padded = False
            # Until row full
            while len(row) < row_capacity:
                # Ensure buffer has conversations
                # Top up
                while len(conv_buffer) < buffer_size:
                    # Refill
                    refill_buffer()

                # Space left
                remaining = row_capacity - len(row)

                # Find largest conversation that fits entirely
                # Best index
                best_idx = -1
                # Best length
                best_len = 0
                # Scan pool
                for i, (conv, _) in enumerate(conv_buffer):
                    # Length
                    conv_len = len(conv)
                    # Fits and longest?
                    if conv_len <= remaining and conv_len > best_len:
                        # Record index
                        best_idx = i
                        # Record length
                        best_len = conv_len

                # Something fits
                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    # Take it
                    conv, conv_mask = conv_buffer.pop(best_idx)
                    # Add tokens
                    row.extend(conv)
                    # Add mask
                    mask_row.extend(conv_mask)
                    # Count consumption
                    consumed += ddp_world_size  # Track actual consumption
                # Nothing fits
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    # Real content length
                    content_len = len(row)
                    # Fill with BOS tokens
                    row.extend([bos_token] * remaining)  # Pad with BOS tokens
                    # Padding is not trained on
                    mask_row.extend([0] * remaining)
                    # Mark
                    padded = True
                    # Row done
                    break  # Row is now full (with padding)

            # Track content length: full row if no padding, otherwise the length before padding
            # Padded row
            if padded:
                # Record real length
                row_lengths.append(content_len)
            # Full row
            else:
                # Record capacity
                row_lengths.append(row_capacity)
            # Store row
            rows.append(row[:row_capacity])
            # Store mask
            mask_rows.append(mask_row[:row_capacity])

        # Stopping condition to respect num_iterations, if given
        # Count batch
        it += 1
        # Reached requested steps?
        if 0 < args.num_iterations <= it and split == "train":
            # Stop after this
            last_step = True

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        # Training split only
        if split == "train":
            # Update epoch
            current_epoch = epoch
            # Fixed steps
            if args.num_iterations > 0:
                # Progress by steps
                approx_progress = it / args.num_iterations
            # One epoch
            else:
                # Progress by data used
                approx_progress = consumed / dataset_size
            # Trigger last_step when we've consumed enough (instead of when cursor wraps)
            # Used everything?
            if consumed >= dataset_size:
                # Stop
                last_step = True

        # Build tensors
        # On CUDA?
        use_cuda = device_type == "cuda"
        # (B, T+1) tensor in pinned memory
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        # Inputs: first T tokens, on GPU
        inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
        # Targets: shifted by one, on GPU
        targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

        # Apply the loss mask from render_conversation (mask=1 for assistant completions,
        # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
        # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
        # Mask tensor
        mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
        # Align with targets
        mask_targets = mask_tensor[:, 1:].to(device=device)
        # -1 = cross-entropy ignores it (contributes zero loss and zero gradient)
        targets[mask_targets == 0] = -1

        # Mask out padding positions in targets (set to -1 = ignore_index)
        # For each row, positions >= (content_length - 1) in targets should be masked
        # For each row
        for i, content_len in enumerate(row_lengths):
            # If padded
            if content_len < row_capacity:
                # Ignore padding targets
                targets[i, content_len-1:] = -1

        # Yield batch
        yield inputs, targets

# Training generator
train_loader = sft_data_generator_bos_bestfit("train")
# Validation generator factory
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
# Progress 0 -> 1
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate schedule (linear warmup, constant, linear warmdown)
# Same shape as base_train but uses progress (0→1) instead of absolute step counts,
# because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
# LR multiplier from progress
def get_lr_multiplier(progress):
    # Warmup
    if progress < args.warmup_ratio:
        # Ramp up
        return (progress + 1e-8) / args.warmup_ratio
    # Constant phase
    elif progress <= 1.0 - args.warmdown_ratio:
        # Full
        return 1.0
    # Warmdown
    else:
        # Fraction through warmdown
        decay = (progress - (1.0 - args.warmdown_ratio)) / args.warmdown_ratio
        # Linear decay to final fraction
        return (1 - decay) * 1.0 + decay * args.final_lr_frac

# Momentum scheduler for Muon optimizer
# Muon momentum by step
def get_muon_momentum(it):
    # Fraction of the first 300 steps
    frac = min(it / 300, 1)
    # Ramp 0.85 -> 0.95
    momentum = (1 - frac) * 0.85 + frac * 0.95
    # Return
    return momentum

# -----------------------------------------------------------------------------
# Training loop
# First batch
x, y = next(train_loader) # prefetch the very first batch of data
# Best val
min_val_bpb = float("inf")
# Smoothed loss
smooth_train_loss = 0 # EMA of training loss
# Smoothing factor
ema_beta = 0.9 # EMA decay factor
# Time
total_training_time = 0 # total wall-clock time of training
# Step
step = 0
# Main loop
while True:
    # Compute so far
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    # Multi-GPU
    if ddp:
        # Flag as tensor
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        # If ANY GPU ran out of data, all stop
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        # Back to bool
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    # Time to eval?
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        # Eval mode
        model.eval()
        # Fresh val loader
        val_loader = build_val_loader()
        # Number of batches
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        # Measure
        val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        # Report
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        # Best?
        if val_bpb < min_val_bpb:
            # Record
            min_val_bpb = val_bpb
        # Log
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
        # Train mode
        model.train()

    # once in a while: estimate the ChatCORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # Results
    chatcore_results = {}
    # Time for ChatCORE?
    if args.chatcore_every > 0 and (last_step or (step > 0 and step % args.chatcore_every == 0)):
        # Eval mode
        model.eval()
        # Engine
        engine = Engine(orig_model, tokenizer)
        # Benchmarks
        all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval']
        # Multiple-choice ones
        categorical_tasks = {'ARC-Easy', 'ARC-Challenge', 'MMLU'}
        # Random-guess accuracy per task
        baseline_accuracies = {
            # 4-way multiple choice = 25%
            'ARC-Easy': 0.25, 'ARC-Challenge': 0.25, 'MMLU': 0.25,
            # Generative = 0%
            'GSM8K': 0.0, 'HumanEval': 0.0,
        # Close
        }
        # Accuracy per task
        task_results = {}
        # For each task
        for task_name in all_tasks:
            # Problem limit
            limit = args.chatcore_max_cat if task_name in categorical_tasks else args.chatcore_max_sample
            # -1 = no limit
            max_problems = None if limit < 0 else limit  # -1 means no limit
            # Run evaluation
            acc = run_chat_eval(task_name, orig_model, tokenizer, engine,
                                # Batch and limit
                                batch_size=args.device_batch_size, max_problems=max_problems)
            # Record
            task_results[task_name] = acc
            # Report
            print0(f"  {task_name}: {100*acc:.2f}%")
        # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
        # Average of (acc - random) / (1 - random)
        def centered_mean(tasks):
            # Formula
            return sum((task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks) / len(tasks)
        # All tasks
        chatcore = centered_mean(all_tasks)
        # Multiple-choice only
        chatcore_cat = centered_mean(categorical_tasks)
        # Report
        print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
        # Log
        wandb_run.log({
            # Step
            "step": step,
            # Compute
            "total_training_flops": flops_so_far,
            # Metric
            "chatcore_metric": chatcore,
            # Metric
            "chatcore_cat": chatcore_cat,
            # Per-task
            **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
        # Close
        })
        # Train mode
        model.train()

    # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
    # At the end
    if last_step:
        # Folder name
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
        # Full path
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        # Save
        save_checkpoint(
            # Folder
            checkpoint_dir,
            # Step
            step,
            # Weights
            orig_model.state_dict(),
            # Optimizer
            optimizer.state_dict(),
            # Metadata
            {
                # Step
                "step": step,
                # Val metric
                "val_bpb": val_bpb, # loss at last step
                # Model shape
                "model_config": {
                    # Context
                    "sequence_len": args.max_seq_len,
                    # Vocab
                    "vocab_size": tokenizer.get_vocab_size(),
                    # Layers
                    "n_layer": depth,
                    # Heads
                    "n_head": model.config.n_head,
                    # KV heads
                    "n_kv_head": model.config.n_kv_head,
                    # Width
                    "n_embd": model.config.n_embd,
                    # Windows
                    "window_pattern": model.config.window_pattern,
                # Close
                },
                # CLI args
                "user_config": user_config, # inputs to the training script
            # Close
            },
            # Rank
            rank=ddp_rank,
        # Close
        )

    # Done?
    if last_step:
        # Exit
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    # Sync for timing
    synchronize()
    # Start timer
    t0 = time.time()
    # Gradient accumulation
    for micro_step in range(grad_accum_steps):
        # Forward: loss on assistant tokens only
        loss = model(x, y)
        # For logging
        train_loss = loss.detach() # for logging
        # Normalize
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        # fp16
        if scaler is not None:
            # Scaled backward
            scaler.scale(loss).backward()
        # bf16
        else:
            # Backward (chain rule)
            loss.backward()
        # Next batch
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        # Update progress (never decreasing)
        progress = max(progress, approx_progress) # only increase progress monotonically
    # step the optimizer
    # LR multiplier
    lrm = get_lr_multiplier(progress)
    # Muon momentum
    muon_momentum = get_muon_momentum(step)
    # For each group
    for group in optimizer.param_groups:
        # Set LR
        group["lr"] = group["initial_lr"] * lrm
        # Muon groups
        if group['kind'] == 'muon':
            # Set momentum
            group["momentum"] = muon_momentum
    # fp16
    if scaler is not None:
        # Unscale gradients
        scaler.unscale_(optimizer)
        # Multi-GPU
        if is_ddp_initialized():
            # For each inf flag
            for v in scaler._found_inf_per_device(optimizer).values():
                # Share
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        # Step
        scaler.step(optimizer)
        # Update scale
        scaler.update()
    # bf16
    else:
        # Gradient-descent update
        optimizer.step()
    # Clear gradients
    model.zero_grad(set_to_none=True)
    # Sync
    synchronize()
    # Stop timer
    t1 = time.time()
    # Duration
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    # Next step
    step += 1

    # logging
    # EMA loss
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    # Debias
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    # Percent done
    pct_done = 100 * progress
    # Throughput
    tok_per_sec = int(args.total_batch_size / dt)
    # FLOPs/sec
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    # Utilization
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    # Skip warm-up steps
    if step > 10:
        # Accumulate time
        total_training_time += dt # only count the time after the first 10 steps
    # Print progress
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    # Every 10 steps
    if step % 10 == 0:
        # Log
        wandb_run.log({
            # Step
            "step": step,
            # Compute
            "total_training_flops": flops_so_far,
            # Time
            "total_training_time": total_training_time,
            # Loss
            "train/loss": debiased_smooth_loss,
            # LR mult
            "train/lrm": lrm,
            # Step time
            "train/dt": dt,
            # Throughput
            "train/tok_per_sec": tok_per_sec,
            # Utilization
            "train/mfu": mfu,
            # Epoch
            "train/epoch": current_epoch,
        # Close
        })

    # The garbage collector spends ~500ms scanning for cycles quite frequently.
    # We manually manage it to avoid these pauses during training.
    # After first step
    if step == 1:
        # Collect setup garbage
        gc.collect() # manually collect a lot of garbage from setup
        # Freeze survivors
        gc.freeze() # freeze all currently surviving objects and exclude them from GC
        # Disable auto GC
        gc.disable() # disable GC entirely except:
    # Occasionally
    elif step % 5000 == 0: # every 5000 steps...
        # Manual collect
        gc.collect() # manually collect, just to be safe for very long runs

# print a few more stats
# Peak memory
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
# Total time
print0(f"Total training time: {total_training_time/60:.2f}m")
# Best val
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# cleanup
# Close wandb
wandb_run.finish() # wandb run finish
# Tear down distributed
compute_cleanup()
