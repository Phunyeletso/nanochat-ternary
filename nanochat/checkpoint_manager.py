"""
Utilities for saving and loading model/optim/state checkpoints.
"""
# STUDY NOTE (added): a "trained model" is just its weight tensors saved to disk. The optimizer state
# (Adam's m and v, Muon's momentum) is saved too so training can resume exactly; it is ~2x the size of the weights.

# Import os
import os
# Import regex
import re
# Import json
import json
# Import logging
import logging
# Import PyTorch
import torch

# Import base dir helper
from nanochat.common import get_base_dir
# Import model classes
from nanochat.gpt import GPT, GPTConfig
# Import tokenizer loader
from nanochat.tokenizer import get_tokenizer
# Import logging setup
from nanochat.common import setup_default_logging

# Set up logging
# Configure logging
setup_default_logging()
# Logger
logger = logging.getLogger(__name__)
# Log only from rank 0
def log0(message):
    # Rank 0?
    if int(os.environ.get('RANK', 0)) == 0:
        # Log
        logger.info(message)

# Fill in config keys that older checkpoints didn't have
def _patch_missing_config_keys(model_config_kwargs):
    """Add default values for new config keys missing in old checkpoints."""
    # Old models were trained with full context (no sliding window)
    # Missing window pattern?
    if "window_pattern" not in model_config_kwargs:
        # Default to full attention
        model_config_kwargs["window_pattern"] = "L"
        # Report
        log0(f"Patching missing window_pattern in model config to 'L'")

# Fill in parameters older checkpoints didn't have
def _patch_missing_keys(model_data, model_config):
    """Add default values for new parameters that may be missing in old checkpoints."""
    # Number of layers
    n_layer = model_config.n_layer
    # resid_lambdas defaults to 1.0 (identity scaling)
    # Missing?
    if "resid_lambdas" not in model_data:
        # Ones = no scaling
        model_data["resid_lambdas"] = torch.ones(n_layer)
        # Report
        log0(f"Patching missing resid_lambdas in model data to 1.0")
    # x0_lambdas defaults to 0.0 (disabled)
    # Missing?
    if "x0_lambdas" not in model_data:
        # Zeros = disabled
        model_data["x0_lambdas"] = torch.zeros(n_layer)
        # Report
        log0(f"Patching missing x0_lambdas in model data to 0.0")

# Save weights, optimizer state and metadata
def save_checkpoint(checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0):
    # Only rank 0 saves the (identical) weights
    if rank == 0:
        # Make folder
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save the model state parameters
        # Weights file path
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        # Save the weights dict
        torch.save(model_data, model_path)
        # Report
        logger.info(f"Saved model parameters to: {model_path}")
        # Save the metadata dict as json
        # Metadata path
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        # Open
        with open(meta_path, "w", encoding="utf-8") as f:
            # Write JSON
            json.dump(meta_data, f, indent=2)
        # Report
        logger.info(f"Saved metadata to: {meta_path}")
    # Note that optimizer state is sharded across ranks, so each rank must save its own.
    # If optimizer state given
    if optimizer_data is not None:
        # Make folder
        os.makedirs(checkpoint_dir, exist_ok=True)
        # One file per GPU (each holds a different shard)
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        # Save
        torch.save(optimizer_data, optimizer_path)
        # Report
        logger.info(f"Saved optimizer state to: {optimizer_path}")

# Load weights, optional optimizer state, and metadata
def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    # Load the model state
    # Weights path
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    # Load onto device
    model_data = torch.load(model_path, map_location=device)
    # Load the optimizer state if requested
    # Default none
    optimizer_data = None
    # If requested
    if load_optimizer:
        # This rank's shard
        optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
        # Load
        optimizer_data = torch.load(optimizer_path, map_location=device)
    # Load the metadata
    # Metadata path
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    # Open
    with open(meta_path, "r", encoding="utf-8") as f:
        # Parse JSON
        meta_data = json.load(f)
    # Return all three
    return model_data, optimizer_data, meta_data


# Rebuild a ready-to-use model from a checkpoint
def build_model(checkpoint_dir, step, device, phase):
    """
    A bunch of repetitive code to build a model from a given checkpoint.
    Returns:
    - base model - uncompiled, not wrapped in DDP
    - tokenizer
    - meta data saved during base model training
    """
    # Validate phase
    assert phase in ["train", "eval"], f"Invalid phase: {phase}"
    # Load files
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)
    # On CPU/Mac...
    if device.type in {"cpu", "mps"}:
        # Convert bfloat16 tensors to float for CPU inference
        # ...convert bf16 weights to fp32
        model_data = {
            # Convert if bf16
            k: v.float() if v.dtype == torch.bfloat16 else v
            # For each entry
            for k, v in model_data.items()
        # Close
        }
    # Hack: fix torch compile issue, which prepends all keys with _orig_mod.
    # Strip the prefix torch.compile adds to names
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    # Saved config
    model_config_kwargs = meta_data["model_config"]
    # Patch old configs
    _patch_missing_config_keys(model_config_kwargs)
    # Report
    log0(f"Building model with config: {model_config_kwargs}")
    # Config object
    model_config = GPTConfig(**model_config_kwargs)
    # Patch old weights
    _patch_missing_keys(model_data, model_config)
    # Shapes only
    with torch.device("meta"):
        # Skeleton
        model = GPT(model_config)
    # Load the model state
    # Allocate memory
    model.to_empty(device=device)
    # Init (needed for rotary tables)
    model.init_weights() # note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
    # Copy in the trained weights
    model.load_state_dict(model_data, strict=True, assign=True)
    # Put the model in the right training phase / mode
    # Eval phase
    if phase == "eval":
        # Eval mode
        model.eval()
    # Train phase
    else:
        # Train mode
        model.train()
    # Load the Tokenizer
    # Tokenizer
    tokenizer = get_tokenizer()
    # Sanity check: compatibility between model and tokenizer
    # Vocab sizes must match (embedding rows = token ids)
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], f"Tokenizer vocab size {tokenizer.get_vocab_size()} does not match model config vocab size {model_config_kwargs['vocab_size']}"
    # Return
    return model, tokenizer, meta_data


# Guess which model folder to use
def find_largest_model(checkpoints_dir):
    # attempt to guess the model tag: take the biggest model available
    # Subfolder names
    model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
    # None?
    if not model_tags:
        # Error
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    # 1) normally all model tags are of the form d<number>, try that first:
    # (depth, tag) pairs
    candidates = []
    # For each tag
    for model_tag in model_tags:
        # Match "d<number>"
        match = re.match(r"d(\d+)", model_tag)
        # If matched
        if match:
            # Extract depth
            model_depth = int(match.group(1))
            # Record
            candidates.append((model_depth, model_tag))
    # If any
    if candidates:
        # Sort by depth descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        # Deepest
        return candidates[0][1]
    # 2) if that failed, take the most recently updated model:
    # Sort by modification time
    model_tags.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    # Newest
    return model_tags[0]


# Find the highest saved step
def find_last_step(checkpoint_dir):
    # Look into checkpoint_dir and find model_<step>.pt with the highest step
    # Weight files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.search(r'model_(\d+)\.pt$', f)]
    # None?
    if not checkpoint_files:
        # Error
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    # Largest step number
    last_step = max(int(f.split("_")[-1].split(".")[0]) for f in checkpoint_files)
    # Return
    return last_step

# -----------------------------------------------------------------------------
# convenience functions that take into account nanochat's directory structure

# Load a model from a checkpoints folder
def load_model_from_dir(checkpoints_dir, device, phase, model_tag=None, step=None):
    # No tag given
    if model_tag is None:
        # guess the model tag by defaulting to the largest model
        # Guess
        model_tag = find_largest_model(checkpoints_dir)
        # Report
        log0(f"No model tag provided, guessing model tag: {model_tag}")
    # Model folder
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    # No step given
    if step is None:
        # guess the step by defaulting to the last step
        # Use the latest
        step = find_last_step(checkpoint_dir)
    # Must have a step
    assert step is not None, f"No checkpoints found in {checkpoint_dir}"
    # build the model
    # Report
    log0(f"Loading model from {checkpoint_dir} with step {step}")
    # Build
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    # Return
    return model, tokenizer, meta_data

# Load by training stage: "base" (pretrained), "sft" (chat fine-tuned), "rl" (reinforcement learned)
def load_model(source, *args, **kwargs):
    # Folder per stage
    model_dir = {
        # Pretraining
        "base": "base_checkpoints",
        # Supervised fine-tuning
        "sft": "chatsft_checkpoints",
        # Reinforcement learning
        "rl": "chatrl_checkpoints",
    # Pick
    }[source]
    # Base dir
    base_dir = get_base_dir()
    # Full path
    checkpoints_dir = os.path.join(base_dir, model_dir)
    # Load
    return load_model_from_dir(checkpoints_dir, *args, **kwargs)

# Load only the optimizer shard
def load_optimizer_state(source, device, rank, model_tag=None, step=None):
    """Load just the optimizer shard for a given rank, without re-loading the model."""
    # Folder per stage
    model_dir = {
        # Pretraining
        "base": "base_checkpoints",
        # SFT
        "sft": "chatsft_checkpoints",
        # RL
        "rl": "chatrl_checkpoints",
    # Pick
    }[source]
    # Base dir
    base_dir = get_base_dir()
    # Path
    checkpoints_dir = os.path.join(base_dir, model_dir)
    # No tag
    if model_tag is None:
        # Guess
        model_tag = find_largest_model(checkpoints_dir)
    # Model folder
    checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    # No step
    if step is None:
        # Latest
        step = find_last_step(checkpoint_dir)
    # Shard path
    optimizer_path = os.path.join(checkpoint_dir, f"optim_{step:06d}_rank{rank:d}.pt")
    # Missing?
    if not os.path.exists(optimizer_path):
        # Report
        log0(f"Optimizer checkpoint not found: {optimizer_path}")
        # Nothing
        return None
    # Report
    log0(f"Loading optimizer state from {optimizer_path}")
    # Load
    optimizer_data = torch.load(optimizer_path, map_location=device)
    # Return
    return optimizer_data
