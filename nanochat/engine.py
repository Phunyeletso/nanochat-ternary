"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""
# STUDY NOTE (added): why a KV cache? Generating token t+1 needs attention over all previous tokens' keys
# and values. Without a cache we'd recompute K and V for the whole sequence every step (O(T^2) total work).
# With the cache, each new token only computes its own q, k, v (one row through every matmul) and reads the
# stored K, V. Inference then becomes memory-bandwidth bound: every step must stream ALL weights from GPU memory.

# Import PyTorch
import torch
# Import functional API (softmax)
import torch.nn.functional as F
# Import signal (for timeouts on the calculator tool)
import signal
# Import warnings
import warnings
# Import contextmanager
from contextlib import contextmanager
# Import deque (fast queue)
from collections import deque
# Import setup helpers and compute dtype
from nanochat.common import compute_init, autodetect_device_type, COMPUTE_DTYPE
# Import model loader
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
# Make it a "with" helper
@contextmanager
# Raise an exception if the block takes longer than `duration` seconds
def timeout(duration, formula):
    # Handler called when the alarm fires
    def timeout_handler(signum, frame):
        # Abort with a message
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    # Register the handler for SIGALRM
    signal.signal(signal.SIGALRM, timeout_handler)
    # Start the alarm
    signal.alarm(duration)
    # Run the block
    yield
    # Cancel the alarm
    signal.alarm(0)

# Evaluate a Python expression with a time limit and no builtins
def eval_with_timeout(formula, max_time=3):
    # Catch any failure
    try:
        # With a timeout
        with timeout(max_time, formula):
            # Suppress warnings
            with warnings.catch_warnings():
                # Ignore syntax warnings
                warnings.simplefilter("ignore", SyntaxWarning)
                # Evaluate with no access to builtins (sandbox-ish)
                return eval(formula, {"__builtins__": {}}, {})
    # On error
    except Exception as e:
        # Cancel alarm
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        # No result
        return None

# The "calculator" tool the model can call inside <|python_start|>...<|python_end|>
def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    # "1,000" -> "1000"
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    # Only digits and operators?
    if all([x in "0123456789*+-/.() " for x in expr]):
        # Power operator could create gigantic numbers
        if "**" in expr:  # disallow power operator
            # Refuse
            return None
        # Evaluate
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    # Allowed characters
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    # Any other character?
    if not all([x in allowed_chars for x in expr]):
        # Refuse
        return None

    # Disallow dangerous patterns
    # Blocklist
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         # More blocked names
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         # More blocked names
                         'getattr', 'setattr', 'delattr', 'hasattr']
    # Lowercase
    expr_lower = expr.lower()
    # Contains anything blocked?
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        # Refuse
        return None

    # Only allow .count() method for now (can expand later)
    # Must be a .count() call
    if '.count(' not in expr:
        # Refuse
        return None

    # Evaluate with timeout
    # Evaluate
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
# Storage for past keys and values during generation
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.

    Key differences from FA2-style cache:
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - Position tracked per batch element via cache_seqlens tensor
    """

    # Allocate the cache
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        # Rows
        self.batch_size = batch_size
        # Max tokens
        self.max_seq_len = seq_len
        # Layers
        self.n_layers = num_layers
        # KV heads
        self.n_heads = num_heads
        # Head dim
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        # Keys for every layer/row/position/head
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Values likewise
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        # Filled length per row
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        # For the "smear" trick
        self.prev_embedding = None

    # Empty the cache
    def reset(self):
        """Reset cache to empty state."""
        # Lengths to zero
        self.cache_seqlens.zero_()
        # Clear smear state
        self.prev_embedding = None

    # Current position
    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        # Row 0's length
        return self.cache_seqlens[0].item()

    # K and V for one layer
    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        # Views (no copy)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    # Move position forward
    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        # Add to every row
        self.cache_seqlens += num_tokens

    # Copy another (batch-1) cache into every row of this one
    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        """
        # Must be empty
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        # Shapes must match
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        # Must be big enough
        assert self.max_seq_len >= other.max_seq_len
        # How many tokens to copy
        other_pos = other.get_pos()
        # Copy keys (broadcast batch 1 -> B)
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        # Copy values
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        # Set lengths
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        # If smear state exists...
        if other.prev_embedding is not None:
            # ...copy it to every row
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
# No autograd during sampling
@torch.inference_mode()
# Pick the next token from logits
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    # Validate
    assert temperature >= 0.0, "temperature must be non-negative"
    # Greedy
    if temperature == 0.0:
        # Highest score
        return torch.argmax(logits, dim=-1, keepdim=True)
    # Top-k sampling
    if top_k is not None and top_k > 0:
        # k can't exceed vocab
        k = min(top_k, logits.size(-1))
        # k best scores and their token ids
        vals, idx = torch.topk(logits, k, dim=-1)
        # Apply temperature
        vals = vals / temperature
        # Probabilities over the k candidates
        probs = F.softmax(vals, dim=-1)
        # Sample one candidate
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        # Map back to token ids
        return idx.gather(1, choice)
    # Full-vocab sampling
    else:
        # Apply temperature
        logits = logits / temperature
        # Probabilities
        probs = F.softmax(logits, dim=-1)
        # Sample
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

# State of one generated sequence
class RowState:
    # Per-row state tracking during generation
    # Constructor
    def __init__(self, current_tokens=None):
        # Tokens so far
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        # Tokens to inject (e.g. calculator output)
        self.forced_tokens = deque() # Queue of tokens to force inject
        # Inside a tool call?
        self.in_python_block = False # Whether we are inside a python block
        # Tool call tokens collected
        self.python_expr_tokens = [] # Tokens of the current python expression
        # Finished?
        self.completed = False # Whether this row has completed generation

# The inference engine
class Engine:

    # Constructor
    def __init__(self, model, tokenizer):
        # The model
        self.model = model
        # Tokenizer (for tool use)
        self.tokenizer = tokenizer # needed for tool use

    # No autograd
    @torch.inference_mode()
    # Streaming generation of num_samples continuations
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        # Validate input
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        # Device
        device = self.model.get_device()
        # Allocate the KV cache in the compute dtype so it matches what the forward pass emits
        # Dtype
        dtype = COMPUTE_DTYPE
        # Random generator
        rng = torch.Generator(device=device)
        # Seed it
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        # Helper
        get_special = lambda s: self.tokenizer.encode_special(s)
        # Tool start
        python_start = get_special("<|python_start|>")
        # Tool end
        python_end = get_special("<|python_end|>")
        # Output start
        output_start = get_special("<|output_start|>")
        # Output end
        output_end = get_special("<|output_end|>")
        # End of answer
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        # BOS also ends
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        # Model config
        m = self.model.config
        # Cache shape arguments
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        # Small cache for the prompt
        kv_cache_prefill = KVCache(
            # One row
            batch_size=1,
            # Prompt length
            seq_len=len(tokens),
            # Device
            device=device,
            # Dtype
            dtype=dtype,
            # Shape args
            **kv_model_kwargs,
        # Close
        )
        # Prompt as tensor
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        # Process the whole prompt in one forward pass (a big matmul-friendly batch: compute bound)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        # Last position's logits, copied for every sample
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        # Size the cache for prompt + answer
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        # Cache for all samples
        kv_cache_decode = KVCache(
            # Rows
            batch_size=num_samples,
            # Length
            seq_len=kv_length_hint,
            # Device
            device=device,
            # Dtype
            dtype=dtype,
            # Shape args
            **kv_model_kwargs,
        # Close
        )
        # Copy the prompt's K/V into every row (compute once, share)
        kv_cache_decode.prefill(kv_cache_prefill)
        # Free memory
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        # One state per row
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        # Counter
        num_generated = 0
        # Loop
        while True:
            # Stop condition: we've reached max tokens
            # Hit limit?
            if max_tokens is not None and num_generated >= max_tokens:
                # Stop
                break
            # Stop condition: all rows are completed
            # All done?
            if all(state.completed for state in row_states):
                # Stop
                break

            # Sample the next token for each row
            # (B, 1) sampled ids
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            # As Python list
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            # Next token per row
            token_column = [] # contains the next token id along each row
            # 1 = sampled by model, 0 = forced by tool
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            # For each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                # Tool output waiting?
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                # Record mask
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                # Forced token or sampled token
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                # Record
                token_column.append(next_token)
                # Update the state of this row to include the next token
                # Append to history
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                # End token?
                if next_token == assistant_end or next_token == bos:
                    # Done
                    state.completed = True
                # Handle tool logic
                # Tool call starts
                if next_token == python_start:
                    # Enter tool mode
                    state.in_python_block = True
                    # Reset expression
                    state.python_expr_tokens = []
                # Tool call ends
                elif next_token == python_end and state.in_python_block:
                    # Leave tool mode
                    state.in_python_block = False
                    # If there's an expression
                    if state.python_expr_tokens:
                        # Decode it
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        # Run the calculator
                        result = use_calculator(expr)
                        # If it worked
                        if result is not None:
                            # Tokenize the result
                            result_tokens = self.tokenizer.encode(str(result))
                            # Queue output start
                            state.forced_tokens.append(output_start)
                            # Queue result
                            state.forced_tokens.extend(result_tokens)
                            # Queue output end
                            state.forced_tokens.append(output_end)
                    # Reset
                    state.python_expr_tokens = []
                # Inside a tool call
                elif state.in_python_block:
                    # Collect the token
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            # Stream back
            yield token_column, token_masks
            # Count
            num_generated += 1

            # Prepare logits for next iteration
            # New tokens as (B, 1)
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            # Forward ONE token per row using the cache (matrix-vector-like work: memory bound)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    # Non-streaming wrapper: return complete sequences
    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        # End token
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        # BOS
        bos = self.tokenizer.get_bos_token_id()
        # Each starts with the prompt
        results = [tokens.copy() for _ in range(num_samples)]
        # Prompt tokens have mask 0
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        # Completion flags
        completed = [False] * num_samples
        # Stream tokens
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            # For each row
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                # Still running?
                if not completed[i]:
                    # End token?
                    if token == assistant_end or token == bos:
                        # Mark done
                        completed[i] = True
                    # Normal token
                    else:
                        # Append token
                        results[i].append(token)
                        # Append mask
                        masks[i].append(mask)
            # Stop if all rows are completed
            # All done?
            if all(completed):
                # Stop
                break
        # Return sequences and masks
        return results, masks


# Self-test when run directly
if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    # Import time
    import time
    # init compute
    # Device
    device_type = autodetect_device_type()
    # Init
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    # Load base model
    model, tokenizer, meta = load_model("base", device, phase="eval")
    # BOS id
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    # Greedy, 64 tokens
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    # Prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    # Output
    generated_tokens = []
    # Sync
    torch.cuda.synchronize()
    # Timer
    t0 = time.time()
    # Naive generator (no cache)
    stream = model.generate(prompt_tokens, **kwargs)
    # For each token
    for token in stream:
        # Record
        generated_tokens.append(token)
        # Decode
        chunk = tokenizer.decode([token])
        # Print
        print(chunk, end="", flush=True)
    # Newline
    print()
    # Sync
    torch.cuda.synchronize()
    # Timer
    t1 = time.time()
    # Report
    print(f"Reference time: {t1 - t0:.2f}s")
    # Save reference
    reference_ids = generated_tokens
    # generate tokens with Engine
    # Output
    generated_tokens = []
    # Engine
    engine = Engine(model, tokenizer)
    # Cached generator
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    # Sync
    torch.cuda.synchronize()
    # Timer
    t0 = time.time()
    # For each step
    for token_column, token_masks in stream:
        # Row 0
        token = token_column[0] # only print out the first row
        # Record
        generated_tokens.append(token)
        # Decode
        chunk = tokenizer.decode([token])
        # Print
        print(chunk, end="", flush=True)
    # Newline
    print()
    # Sync
    torch.cuda.synchronize()
    # Timer
    t1 = time.time()
    # Report
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    # For each position
    for i in range(len(reference_ids)):
        # Different?
        if reference_ids[i] != generated_tokens[i]:
            # Report
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            # Stop
            break
    # Overall match
    print(f"Match: {reference_ids == generated_tokens}")
