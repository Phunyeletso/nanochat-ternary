"""
A number of functions that help with evaluating a base model.
"""
# STUDY NOTE (added): cross-entropy loss is measured in "nats" (natural log). Dividing by ln(2) converts to bits.
# Bits per byte = how many bits the model needs, on average, to encode each byte of text. Lower = better predictor.
# It doesn't depend on the tokenizer, so you can compare models with different vocabularies.

# Import math (for log)
import math
# Import PyTorch
import torch
# Import distributed communication
import torch.distributed as dist

# No gradients needed during evaluation (saves memory and compute: no backward pass)
@torch.no_grad()
# Compute bits-per-byte on `steps` batches
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    # Running sum of loss (in nats)
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    # Running sum of bytes
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    # Iterator over batches
    batch_iter = iter(batches)
    # For each evaluation batch
    for _ in range(steps):
        # Get inputs and targets
        x, y = next(batch_iter)
        # Per-token loss (no averaging), shape (B, T)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        # Flatten to 1D
        loss2d = loss2d.view(-1) # flatten
        # Flatten targets
        y = y.view(-1) # flatten
        # Are there any ignored targets (negative ids)?
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            # Mask of valid targets
            valid = y >= 0
            # Replace invalid ids with 0 so indexing is safe
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            # Byte length per target (0 for ignored)
            num_bytes2d = torch.where(
                # Condition
                valid,
                # Byte length lookup
                token_bytes[y_safe],
                # Zero for ignored
                torch.zeros_like(y, dtype=token_bytes.dtype)
            # Close
            )
            # Add loss only for tokens that have bytes
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            # Add bytes
            total_bytes += num_bytes2d.sum()
        # No ignored targets
        else:
            # fast path: no ignored targets, safe to index directly
            # Byte length per target
            num_bytes2d = token_bytes[y]
            # Add loss for counted tokens
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            # Add bytes
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    # Number of GPUs
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    # If several GPUs...
    if world_size > 1:
        # ...sum loss across GPUs
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        # ...sum bytes across GPUs
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    # To Python float
    total_nats = total_nats.item()
    # To Python int
    total_bytes = total_bytes.item()
    # Avoid dividing by zero
    if total_bytes == 0:
        # Infinite = undefined
        return float('inf')
    # nats -> bits (divide by ln 2), then per byte
    bpb = total_nats / (math.log(2) * total_bytes)
    # Return result
    return bpb
