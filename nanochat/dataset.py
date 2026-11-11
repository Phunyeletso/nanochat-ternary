"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""
# STUDY NOTE (added): pretraining data = a huge pile of web text, split into ~6,500 "shard" files.
# The model never sees labels; the text itself is the label (predict the next token). More data -> more
# training tokens -> more FLOPs, because every token costs ~6 x params FLOPs to train on.

# Import os for files
import os
# Import argparse for command-line flags
import argparse
# Import time for retry waits
import time
# Import requests for HTTP downloads
import requests
# Import the Parquet reader
import pyarrow.parquet as pq
# Import a process pool for parallel downloads
from multiprocessing import Pool

# Import base dir helper
from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
# Hugging Face dataset location
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
# Index of the last shard
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
# Function: shard index -> file name (zero-padded to 5 digits)
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
# Base dir
base_dir = get_base_dir()
# Local folder for the data
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

# List downloaded shard files
def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """ Looks into a data dir and returns full paths to all parquet files. """
    # Default folder
    data_dir = DATA_DIR if data_dir is None else data_dir

    # Legacy-supporting code due to the upgrade from FinewebEdu-100B to ClimbMix-400B
    # This code will eventually be deleted.
    # If the new folder doesn't exist...
    if not os.path.exists(data_dir):
        # ...optionally warn
        if warn_on_legacy:
            # Blank line
            print()
            # Separator
            print("=" * 80)
            # Title
            print("  WARNING: DATASET UPGRADE REQUIRED")
            # Separator
            print("=" * 80)
            # Blank
            print()
            # Missing folder
            print(f"  Could not find: {data_dir}")
            # Blank
            print()
            # Explanation
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            # Explanation
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            # Explanation
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            # Blank
            print()
            # Command 1
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            # Command 2
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            # Blank
            print()
            # Fallback note
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            # Separator
            print("=" * 80)
            # Blank
            print()
        # attempt a fallback to the legacy data directory
        # Old folder
        data_dir = os.path.join(base_dir, "base_data")

    # Sorted list of complete .parquet files
    parquet_files = sorted([
        # Each file name
        f for f in os.listdir(data_dir)
        # Only finished parquet files (not partial .tmp downloads)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    # Close
    ])
    # Full paths
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    # Return
    return parquet_paths

# Iterate over documents in batches (used e.g. for tokenizer training)
def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    # Validate split
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    # All files
    parquet_paths = list_parquet_files()
    # Train = all but last; val = last
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    # For each file
    for filepath in parquet_paths:
        # Open
        pf = pq.ParquetFile(filepath)
        # For every step-th row group starting at start
        for rg_idx in range(start, pf.num_row_groups, step):
            # Read it
            rg = pf.read_row_group(rg_idx)
            # Text column as Python list
            texts = rg.column('text').to_pylist()
            # Yield the list of documents
            yield texts

# -----------------------------------------------------------------------------
# Download one shard
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    # File name
    filename = index_to_filename(index)
    # Local path
    filepath = os.path.join(DATA_DIR, filename)
    # Already downloaded?
    if os.path.exists(filepath):
        # Skip
        print(f"Skipping {filepath} (already exists)")
        # Success
        return True

    # Construct the remote URL for this file
    # Remote URL
    url = f"{BASE_URL}/{filename}"
    # Report
    print(f"Downloading {filename}...")

    # Download with retries
    # Max tries
    max_attempts = 5
    # Try up to 5 times
    for attempt in range(1, max_attempts + 1):
        # Catch network errors
        try:
            # Start streaming download
            response = requests.get(url, stream=True, timeout=30)
            # Raise on HTTP errors (404, 500, ...)
            response.raise_for_status()
            # Write to temporary file first
            # Temp path
            temp_path = filepath + f".tmp"
            # Open temp file
            with open(temp_path, 'wb') as f:
                # Read 1 MB at a time
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    # Non-empty chunk
                    if chunk:
                        # Write it
                        f.write(chunk)
            # Move temp file to final location
            # Rename atomically once complete
            os.rename(temp_path, filepath)
            # Report
            print(f"Successfully downloaded {filename}")
            # Success
            return True

        # On failure
        except (requests.RequestException, IOError) as e:
            # Report
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            # For temp and final paths
            for path in [filepath + f".tmp", filepath]:
                # If it exists
                if os.path.exists(path):
                    # Try deleting
                    try:
                        # Delete
                        os.remove(path)
                    # Ignore errors
                    except:
                        # Nothing
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            # More attempts left
            if attempt < max_attempts:
                # Wait 2, 4, 8, 16 seconds
                wait_time = 2 ** attempt
                # Report
                print(f"Waiting {wait_time} seconds before retry...")
                # Sleep
                time.sleep(wait_time)
            # Out of attempts
            else:
                # Report
                print(f"Failed to download {filename} after {max_attempts} attempts")
                # Failure
                return False

    # Failure
    return False


# Run when executed as `python -m nanochat.dataset`
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    # How many training shards
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    # Parallel workers
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    # Parse
    args = parser.parse_args()

    # Prepare the output directory
    # Create folder
    os.makedirs(DATA_DIR, exist_ok=True)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    # Number of train shards
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    # Shard indices to fetch
    ids_to_download = list(range(num_train_shards))
    # Plus the validation shard
    ids_to_download.append(MAX_SHARD) # always download the validation shard

    # Download the shards
    # Report
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    # Report
    print(f"Target directory: {DATA_DIR}")
    # Blank
    print()
    # Parallel worker processes
    with Pool(processes=args.num_workers) as pool:
        # Download all shards in parallel
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    # Count successes
    successful = sum(1 for success in results if success)
    # Report
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
