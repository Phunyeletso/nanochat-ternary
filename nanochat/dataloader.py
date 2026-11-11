"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""
# STUDY NOTE (added): the model is trained to predict token t+1 from tokens 0..t.
# So each batch is a (B, T+1) block of tokens: inputs = first T columns, targets = last T columns (shifted by one).
# One (B, T) batch therefore gives B*T separate "predict the next token" training examples at once.

# Import PyTorch
import torch
# Import the Parquet reader (the dataset is stored as compressed columnar Parquet files)
import pyarrow.parquet as pq

# Import info about multi-GPU setup
from nanochat.common import get_dist_info
# Import the function that lists downloaded data shards
from nanochat.dataset import list_parquet_files

# Generator yielding batches of raw text documents forever
def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    # Which GPU am I and how many are there
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    # Only GPU 0 on the train split prints legacy-format warnings
    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    # List all data shard files
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    # Make sure data was downloaded
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    # All but the last shard are training data; the last shard is validation data
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    # File index to resume from
    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    # Row-group index to resume from
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    # Epoch to resume from
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    # First pass through the data?
    first_pass = True
    # Current file index
    pq_idx = resume_pq_idx
    # Current epoch
    epoch = resume_epoch

    # Loop over the dataset forever
    while True:  # iterate infinitely (multi-epoch)
        # Start at the resume file on the first pass, else from file 0
        pq_idx = resume_pq_idx if first_pass else 0
        # For each file
        while pq_idx < len(parquet_paths):
            # File path
            filepath = parquet_paths[pq_idx]
            # Open the parquet file
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            # If resuming inside this file...
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                # Which "round" of row groups we were in
                base_idx = resume_rg_idx // ddp_world_size
                # Skip to the next round
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                # This GPU's row group in that round
                rg_idx = base_idx * ddp_world_size + ddp_rank
                # If past the end of the file...
                if rg_idx >= pf.num_row_groups:
                    # ...go to the next file
                    pq_idx += 1
                    # Restart loop
                    continue
                # Don't apply resume logic again
                resume_rg_idx = None  # only do this once
            # Normal case
            else:
                # Each GPU starts at a different row group (GPU i reads groups i, i+N, i+2N, ...)
                rg_idx = ddp_rank
            # For each of this GPU's row groups
            while rg_idx < pf.num_row_groups:
                # Read the row group (a chunk of documents)
                rg = pf.read_row_group(rg_idx)
                # Extract the "text" column as a Python list of strings
                batch = rg.column('text').to_pylist()
                # Split into small batches for the tokenizer
                for i in range(0, len(batch), tokenizer_batch_size):
                    # Yield the text batch plus the position (for resuming)
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                # Jump N row groups ahead (stride by number of GPUs)
                rg_idx += ddp_world_size
            # Next file
            pq_idx += 1
        # After the first full pass...
        first_pass = False
        # ...increment epoch
        epoch += 1


# The main training data loader: yields (inputs, targets) tensors of shape (B, T)
def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    # Tokenizer, batch rows B, context length T, which split
    tokenizer, B, T, split,
    # Tokenizer parallelism
    tokenizer_threads=4, tokenizer_batch_size=128,
    # Target device and resume info
    device="cuda", resume_state_dict=None,
    # How many tokenized docs to keep around for packing
    buffer_size=1000
# Close signature
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    # Validate the split name
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # Each row needs T+1 tokens (T inputs + 1 extra so targets can be shifted)
    row_capacity = T + 1
    # Raw text document generator
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    # Id of the beginning-of-sequence token
    bos_token = tokenizer.get_bos_token_id()
    # Pool of tokenized documents waiting to be packed
    doc_buffer = []
    # Current data position
    pq_idx, rg_idx, epoch = 0, 0, 1

    # Tokenize another batch of documents and add them to the pool
    def refill_buffer():
        # Allow modifying the outer variables
        nonlocal pq_idx, rg_idx, epoch
        # Get the next text batch and its position
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        # Convert each document's text into a list of token ids, prefixed with BOS
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        # For each tokenized document...
        for tokens in token_lists:
            # ...add to the pool
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    # Are we on an NVIDIA GPU?
    use_cuda = device == "cuda"
    # CPU scratch area for building the (B, T+1) block
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    # Pinned (page-locked) CPU memory allows fast asynchronous copies to the GPU
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    # Matching buffer on the GPU
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    # View of the first half as (B, T) inputs
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    # View of the second half as (B, T) targets
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    # GPU inputs view
    inputs = gpu_buffer[:B * T].view(B, T)
    # GPU targets view
    targets = gpu_buffer[B * T:].view(B, T)

    # Produce batches forever
    while True:
        # Fill each of the B rows
        for row_idx in range(B):
            # Current fill position in this row
            pos = 0
            # Until the row is full
            while pos < row_capacity:
                # Ensure buffer has documents
                # Keep the pool topped up
                while len(doc_buffer) < buffer_size:
                    # Add more documents
                    refill_buffer()

                # Free space left in the row
                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                # Best candidate index
                best_idx = -1
                # Best candidate length
                best_len = 0
                # Scan the pool
                for i, doc in enumerate(doc_buffer):
                    # Document length
                    doc_len = len(doc)
                    # Fits and is the longest so far?
                    if doc_len <= remaining and doc_len > best_len:
                        # Remember index
                        best_idx = i
                        # Remember length
                        best_len = doc_len

                # Found a document that fits
                if best_idx >= 0:
                    # Take it out of the pool
                    doc = doc_buffer.pop(best_idx)
                    # Its length
                    doc_len = len(doc)
                    # Copy its tokens into the row
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    # Advance position
                    pos += doc_len
                # Nothing fits
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    # Index of the shortest document
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    # Take it out
                    doc = doc_buffer.pop(shortest_idx)
                    # Copy just enough of its start to fill the row exactly
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    # Row is now full
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        # Inputs = tokens 0..T-1 of each row
        cpu_inputs.copy_(row_buffer[:, :-1])
        # Targets = tokens 1..T (the "next token" for each input position)
        cpu_targets.copy_(row_buffer[:, 1:])

        # Position info for resuming
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        # One copy CPU -> GPU (non_blocking lets the CPU continue while it transfers)
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        # Hand back the batch
        yield inputs, targets, state_dict

# Same loader, but without the resume state
def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    # Iterate the full loader
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        # Yield only inputs and targets
        yield inputs, targets
