"""
Common utilities for nanochat.
"""

# Import os for env vars and paths
import os
# Import regex
import re
# Import logging
import logging
# Import urllib for simple downloads
import urllib.request
# Import PyTorch
import torch
# Import distributed communication
import torch.distributed as dist
# Import a cross-process file lock
from filelock import FileLock

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
# STUDY NOTE (added): bf16 = 16-bit float with fp32's exponent range but only 8 bits of precision.
# Half the bytes of fp32 => half the memory traffic, and GPU tensor cores multiply bf16 matrices ~10x+ faster than fp32.
# Map from name to torch dtype
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
# Pick the best compute dtype for this hardware
def _detect_compute_dtype():
    # User override via environment variable
    env = os.environ.get("NANOCHAT_DTYPE")
    # If set...
    if env is not None:
        # ...use it
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    # NVIDIA GPU present
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        # GPU architecture version
        capability = torch.cuda.get_device_capability()
        # Ampere or newer
        if capability >= (8, 0):
            # Use bf16
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        # Older GPU: fp32
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    # Note: MPS on recent macOS also handles bf16 fine, opt in via NANOCHAT_DTYPE=bfloat16
    # CPU or Mac: fp32
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
# Decide once at import time
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

# Log formatter that adds terminal colours
class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    # Colour per log level
    COLORS = {
        # Cyan
        'DEBUG': '\033[36m',    # Cyan
        # Green
        'INFO': '\033[32m',     # Green
        # Yellow
        'WARNING': '\033[33m',  # Yellow
        # Red
        'ERROR': '\033[31m',    # Red
        # Magenta
        'CRITICAL': '\033[35m', # Magenta
    # Close
    }
    # Reset code
    RESET = '\033[0m'
    # Bold code
    BOLD = '\033[1m'
    # Format one log record
    def format(self, record):
        # Add color to the level name
        # Original level name
        levelname = record.levelname
        # If it has a colour...
        if levelname in self.COLORS:
            # ...wrap it in colour codes
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        # Standard formatting
        message = super().format(record)
        # Add color to specific parts of the message
        # For INFO messages...
        if levelname == 'INFO':
            # Highlight numbers and percentages
            # Bold sizes/percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            # Highlight "Shard N"
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        # Return formatted text
        return message

# Configure logging once
def setup_default_logging():
    # Log to the terminal
    handler = logging.StreamHandler()
    # Use the coloured formatter
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # Global logging config
    logging.basicConfig(
        # Show INFO and above
        level=logging.INFO,
        # Our handler
        handlers=[handler]
    # Close
    )

# Run setup at import
setup_default_logging()
# Logger for this module
logger = logging.getLogger(__name__)

# Folder where nanochat stores data, tokenizer and checkpoints
def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    # User override
    if os.environ.get("NANOCHAT_BASE_DIR"):
        # Use it
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    # Default
    else:
        # Home folder
        home_dir = os.path.expanduser("~")
        # ~/.cache
        cache_dir = os.path.join(home_dir, ".cache")
        # ~/.cache/nanochat
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    # Create if missing
    os.makedirs(nanochat_dir, exist_ok=True)
    # Return path
    return nanochat_dir

# Download a file once, even if several GPU processes ask at the same time
def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    # Base dir
    base_dir = get_base_dir()
    # Target path
    file_path = os.path.join(base_dir, filename)
    # Lock file path
    lock_path = file_path + ".lock"

    # Already there?
    if os.path.exists(file_path):
        # Done
        return file_path

    # Acquire the lock
    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        # Another process may have downloaded it meanwhile
        if os.path.exists(file_path):
            # Done
            return file_path

        # Download the content as bytes
        # Report
        print(f"Downloading {url}...")
        # Open URL
        with urllib.request.urlopen(url) as response:
            # Read all bytes
            content = response.read() # bytes

        # Write to local file
        # Open for writing
        with open(file_path, 'wb') as f:
            # Write bytes
            f.write(content)
        # Report
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        # Optional post-processing
        if postprocess_fn is not None:
            # Run it
            postprocess_fn(file_path)

    # Return path
    return file_path

# print() that only runs on GPU 0 (avoids 8 copies of every message)
def print0(s="",**kwargs):
    # This process's rank
    ddp_rank = int(os.environ.get('RANK', 0))
    # Only rank 0...
    if ddp_rank == 0:
        # ...prints
        print(s, **kwargs)

# Print the ASCII banner
def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    # The banner text
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    # Print it
    print0(banner)

# Was this script launched with torchrun (multi-GPU)?
def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    # torchrun sets these three env vars
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

# Is multi-GPU communication already set up?
def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    Used at cleanup to avoid destroying a non-existent PG.
    """
    # Check
    return dist.is_available() and dist.is_initialized()

# Return (is_distributed, global rank, local rank, world size)
def get_dist_info():
    # Under torchrun
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        # Sanity check
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        # Global index of this process
        ddp_rank = int(os.environ['RANK'])
        # Index on this machine (= which GPU)
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        # Total processes
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        # Return
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    # Single process
    else:
        # Defaults
        return False, 0, 0, 1

# Pick the best available hardware
def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    # NVIDIA GPU
    if torch.cuda.is_available():
        # cuda
        device_type = "cuda"
    # Apple GPU
    elif torch.backends.mps.is_available():
        # mps
        device_type = "mps"
    # Nothing else
    else:
        # cpu
        device_type = "cpu"
    # Report
    print0(f"Autodetected device type: {device_type}")
    # Return
    return device_type

# Common startup: seeds, precision, multi-GPU setup
def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    # Validate
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    # CUDA requested...
    if device_type == "cuda":
        # ...must be available
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    # MPS requested...
    if device_type == "mps":
        # ...must be available
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    # Fix the random seed (same initial weights every run)
    torch.manual_seed(42)
    # On GPU...
    if device_type == "cuda":
        # ...also seed the GPU RNG
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    # On GPU...
    if device_type == "cuda":
        # ...let fp32 matmuls use TF32 tensor cores (faster, slightly less precise)
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    # Read distributed info
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    # Multi-GPU on CUDA
    if is_ddp_requested and device_type == "cuda":
        # This process uses GPU number local_rank
        device = torch.device("cuda", ddp_local_rank)
        # Make it the default GPU
        torch.cuda.set_device(device)  # make "cuda" default to this device
        # Connect all processes via NCCL (NVIDIA's fast GPU-to-GPU communication library)
        dist.init_process_group(backend="nccl", device_id=device)
        # Wait until all processes are ready
        dist.barrier()
    # Single device
    else:
        # Plain device
        device = torch.device(device_type) # mps|cpu

    # Rank 0...
    if ddp_rank == 0:
        # ...logs world size
        logger.info(f"Distributed world size: {ddp_world_size}")

    # Return info
    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

# Shut down multi-GPU communication
def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    # If set up...
    if is_ddp_initialized():
        # ...tear down
        dist.destroy_process_group()

# A logger that does nothing (same methods as wandb)
class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    # Constructor
    def __init__(self):
        # Nothing
        pass
    # log() does nothing
    def log(self, *args, **kwargs):
        # Nothing
        pass
    # finish() does nothing
    def finish(self):
        # Nothing
        pass

# hardcoded BF16 peak flops for various GPUs
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
# STUDY NOTE (added): training time ~= total FLOPs / (peak FLOPs/s * MFU). E.g. 4e19 FLOPs on 8xH100 at 40% MFU
# = 4e19 / (8 * 989e12 * 0.4) ~= 12,600 s ~= 3.5 hours.
# Theoretical peak bf16 matmul throughput of a GPU (FLOPs per second)
def get_peak_flops(device_name: str) -> float:
    # Lowercase name for matching
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    # (substrings that must all appear, peak FLOPs/s)
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        # GB200
        (["gb200"], 2.5e15),
        # Grace Blackwell
        (["grace blackwell"], 2.5e15),
        # B200
        (["b200"], 2.25e15),
        # B100
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        # H200 NVL
        (["h200", "nvl"], 836e12),
        # H200 PCIe
        (["h200", "pcie"], 836e12),
        # H200 SXM
        (["h200"], 989e12),
        # H100 NVL
        (["h100", "nvl"], 835e12),
        # H100 PCIe
        (["h100", "pcie"], 756e12),
        # H100 SXM (~1 quadrillion bf16 multiply-adds... per second)
        (["h100"], 989e12),
        # H800 NVL
        (["h800", "nvl"], 989e12),
        # H800
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        # A100
        (["a100"], 312e12),
        # A800
        (["a800"], 312e12),
        # A40
        (["a40"], 149.7e12),
        # A30
        (["a30"], 165e12),
        # NVIDIA Ada data center
        # L40S
        (["l40s"], 362e12),
        # L40S alternate spelling
        (["l40-s"], 362e12),
        # L40S alternate spelling
        (["l40 s"], 362e12),
        # L4
        (["l4"], 121e12),
        # AMD CDNA accelerators
        # MI355
        (["mi355"], 2.5e15),
        # MI325
        (["mi325"], 1.3074e15),
        # MI300X
        (["mi300x"], 1.3074e15),
        # MI300A
        (["mi300a"], 980.6e12),
        # MI250X
        (["mi250x"], 383e12),
        # MI250
        (["mi250"], 362.1e12),
        # Consumer RTX
        # RTX 5090
        (["5090"], 209.5e12),
        # RTX 4090
        (["4090"], 165.2e12),
        # RTX 3090
        (["3090"], 71e12),
    # Close
    )
    # Find the first matching row
    for patterns, flops in _PEAK_FLOPS_TABLE:
        # All substrings present?
        if all(p in name for p in patterns):
            # Return its peak
            return flops
    # Intel Ponte Vecchio
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        # Number of compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        # Compute peak from units x clock
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    # Warn
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    # Infinity => MFU 0%
    return float('inf')

# Peak memory bandwidth (bytes/s): limits how fast weights can be streamed in during token-by-token generation
def get_peak_bandwidth(device_name: str) -> float:
    """Peak HBM/GDDR memory bandwidth in bytes/sec. The decode phase of inference
    is memory-bandwidth-bound, so this is the roofline for tokens/sec (see MBU)."""
    # Lowercase
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    # (substrings, bytes per second)
    _PEAK_BANDWIDTH_TABLE = (
        # NVIDIA Blackwell (HBM3e)
        # GB200
        (["gb200"], 8.0e12),
        # Grace Blackwell
        (["grace blackwell"], 8.0e12),
        # B200
        (["b200"], 8.0e12),
        # B100
        (["b100"], 8.0e12),
        # NVIDIA Hopper
        # H200
        (["h200"], 4.8e12),
        # H100 NVL
        (["h100", "nvl"], 3.9e12),
        # H100 PCIe
        (["h100", "pcie"], 2.0e12),
        # H100 SXM
        (["h100"], 3.35e12), # SXM
        # H800 PCIe
        (["h800", "pcie"], 2.0e12),
        # H800 SXM
        (["h800"], 3.35e12), # SXM
        # NVIDIA Ampere data center (A100 80GB; the 40GB variant is 1.6e12)
        # A100
        (["a100"], 2.0e12),
        # A800
        (["a800"], 2.0e12),
        # A40
        (["a40"], 696e9),
        # A30
        (["a30"], 933e9),
        # NVIDIA Ada data center
        # L40S
        (["l40s"], 864e9),
        # L40S alt
        (["l40-s"], 864e9),
        # L40S alt
        (["l40 s"], 864e9),
        # L4
        (["l4"], 300e9),
        # AMD CDNA accelerators
        # MI355
        (["mi355"], 8.0e12),
        # MI325
        (["mi325"], 6.0e12),
        # MI300X
        (["mi300x"], 5.3e12),
        # MI300A
        (["mi300a"], 5.3e12),
        # MI250X
        (["mi250x"], 3.28e12),
        # MI250
        (["mi250"], 3.28e12),
        # Consumer RTX
        # RTX 5090
        (["5090"], 1.79e12),
        # RTX 4090
        (["4090"], 1.01e12),
        # RTX 3090
        (["3090"], 936e9),
    # Close
    )
    # Find first match
    for patterns, bandwidth in _PEAK_BANDWIDTH_TABLE:
        # All substrings present?
        if all(p in name for p in patterns):
            # Return bandwidth
            return bandwidth

    # Unknown GPU - return inf so MBU shows as 0% rather than a wrong guess
    # Warn
    logger.warning(f"Peak bandwidth undefined for: {device_name}, MBU will show as 0%")
    # Infinity
    return float('inf')
