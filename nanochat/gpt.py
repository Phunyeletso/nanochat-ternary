"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""
# =====================================================================================
# STUDY NOTES (added): Notation used in the comments below
#   B = batch size (how many sequences are processed at once)
#   T = sequence length (how many tokens per sequence)
#   C = n_embd = model width / "d_model" (length of the vector that represents each token)
#   H = number of attention heads, D = head_dim = C / H
#   V = vocabulary size (how many distinct tokens exist)
# A "Linear" layer is just a matrix multiply: y = x @ W^T, with x:(B,T,C_in), W:(C_out,C_in).
# Cost of one such matmul = 2 * B*T * C_in * C_out FLOPs (one multiply + one add per weight per token).
# Almost all the compute of an LLM lives in these matmuls, which is why training is so expensive.
# =====================================================================================

# Import `partial`, a helper that pre-fills some arguments of a function (unused here, kept from original)
from functools import partial
# Import `dataclass`, a decorator that auto-generates __init__ etc. for simple config classes
from dataclasses import dataclass

# Import PyTorch: the tensor (n-dimensional array) library with automatic differentiation (autograd)
import torch
# Import the neural-network module namespace (layers like Embedding, Linear, Module base class)
import torch.nn as nn
# Import the functional API (stateless ops like softmax, cross_entropy, rms_norm, linear)
import torch.nn.functional as F

# Import helpers: distributed-training info, a print that only runs on GPU rank 0, and the dtype used for math (e.g. bfloat16)
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
# Import the custom optimizer that combines Muon (for 2D weight matrices) with AdamW (for everything else)
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
# Import the attention kernel wrapper: computes softmax(QK^T/sqrt(D)) V without materialising the full TxT matrix in memory
from nanochat.flash_attention import flash_attn

# Decorator: turn GPTConfig into a simple data-holding class with defaults
@dataclass
# Define the hyperparameters that fully describe the model's shape (and therefore its compute cost)
class GPTConfig:
    # Max number of tokens the model looks at at once (context length T). Attention cost grows with T.
    sequence_len: int = 2048
    # Number of distinct tokens. Sets the size of the embedding table and the final (C x V) output matrix.
    vocab_size: int = 32768
    # Number of stacked Transformer blocks (depth). Params and FLOPs grow linearly with this.
    n_layer: int = 12
    # Number of query heads: attention is split into H parallel smaller attentions of size D = C/H
    n_head: int = 6 # number of query heads
    # Number of key/value heads. If < n_head, several query heads share one K/V head (GQA) -> smaller KV cache
    n_kv_head: int = 6 # number of key/value heads (GQA)
    # Model width C: the length of each token's vector. Params grow ~ C^2, so this is the biggest compute lever.
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    # Pattern of attention window sizes per layer; short windows save attention FLOPs (cost ~ T * window)
    window_pattern: str = "SSSL"


# Define RMSNorm without learnable scale: x / sqrt(mean(x^2)). Keeps vector magnitudes stable so gradients don't explode/vanish.
def norm(x):
    # Normalize over the last dimension (the C features of each token) using root-mean-square
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

# Define a Linear layer (a matrix multiply) that runs in low precision but stores its weights in fp32
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    # Forward pass: given input x of shape (..., in_features), return x @ W^T of shape (..., out_features)
    def forward(self, x):
        # Cast W to x's dtype (e.g. bf16, which GPU tensor cores multiply much faster) and do the matmul y = x W^T
        return F.linear(x, self.weight.to(dtype=x.dtype))


# Decide whether a given layer gets a "value embedding" (an extra lookup table added into the attention values)
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    # Alternate layers: True when layer_idx has the same parity (odd/even) as the last layer
    return layer_idx % 2 == (n_layer - 1) % 2

# Rotary Positional Embedding (RoPE): encode position by ROTATING pairs of features by a position-dependent angle
def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    # Check x is 4D: (B, T, H, D)
    assert x.ndim == 4  # multihead attention
    # Half the head dimension: features are paired (i, i+d) and each pair is rotated like a 2D vector
    d = x.shape[3] // 2
    # Split each head vector into first half x1 and second half x2 (the two coordinates of each 2D pair)
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    # Apply a 2x2 rotation matrix [[cos, sin], [-sin, cos]] to each pair: first output coordinate
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    # Second output coordinate of the rotation. Key fact: dot(rot(q,m), rot(k,n)) depends only on (m - n) = relative position
    y2 = x1 * (-sin) + x2 * cos
    # Glue the two rotated halves back together along the head dimension
    return torch.cat([y1, y2], 3)

# Self-attention: each token builds a weighted average of earlier tokens' information
class CausalSelfAttention(nn.Module):
    # Constructor: create the weight matrices for this attention layer
    def __init__(self, config, layer_idx):
        # Initialize the nn.Module base class (registers parameters for autograd/optimizer)
        super().__init__()
        # Remember which layer this is (needed for KV cache lookup)
        self.layer_idx = layer_idx
        # Number of query heads H
        self.n_head = config.n_head
        # Number of key/value heads
        self.n_kv_head = config.n_kv_head
        # Model width C
        self.n_embd = config.n_embd
        # Per-head width D = C / H
        self.head_dim = self.n_embd // self.n_head
        # C must split evenly into H heads
        assert self.n_embd % self.n_head == 0
        # Query heads must be an integer multiple of KV heads (for GQA sharing)
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # W_q: (C -> H*D) matrix that turns each token vector into its "query" ("what am I looking for?")
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # W_k: (C -> Hkv*D) matrix producing "keys" ("what do I contain?")
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # W_v: (C -> Hkv*D) matrix producing "values" ("what information do I pass on?")
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # W_o: (C -> C) output projection that mixes the concatenated heads back into the residual stream
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # Use only the first 12 channels of x to compute the value-embedding gate (cheap)
        self.ve_gate_channels = 12
        # Tiny (12 -> Hkv) matrix producing one gate value per KV head, only on layers that use value embeddings
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    # Forward pass of attention. x: (B,T,C), ve: optional value embedding, cos_sin: RoPE tables
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Read the three dimensions of the input tensor
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        # Q = x W_q^T  (matmul: 2*B*T*C*C FLOPs), then reshape the last dim into H heads of size D
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        # K = x W_k^T, reshaped into KV heads
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        # V = x W_v^T, reshaped into KV heads
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        # Only some layers have a value embedding
        if ve is not None:
            # Reshape the looked-up value embedding to match V's head layout
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # Gate in (0,3): sigmoid squashes to (0,1), times 3. Small matmul on the first 12 channels of x.
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            # Add the gated embedding to V (unsqueeze makes gate broadcast over the D features)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        # Unpack the precomputed cos and sin tables for the current positions
        cos, sin = cos_sin
        # Rotate Q and K by position-dependent angles so their dot product encodes relative distance
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # Normalize Q and K per head so the dot products q.k stay in a bounded range (stabilizes softmax and training)
        q, k = norm(q), norm(k) # QK norm
        # Scale Q up a bit: bigger dot products -> sharper (more peaked) softmax attention
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        # Scale K by the same factor (total effect on q.k is 1.44x)
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        # No cache means training (whole sequence at once)
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            # y = softmax(Q K^T / sqrt(D) + causal_mask) V. Two matmuls of cost ~2*B*H*T*T*D each -> quadratic in T
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Otherwise we are generating text token by token
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            # Fetch this layer's stored keys and values from previous tokens (so we don't recompute them)
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            # Attention of the new query/queries against all cached keys/values (and append new k,v to the cache)
            y = flash_attn.flash_attn_with_kvcache(
                # The new queries plus the cache buffers
                q, k_cache, v_cache,
                # The new keys/values to insert into the cache
                k=k, v=v,
                # How many tokens are already in the cache for each batch row
                cache_seqlens=kv_cache.cache_seqlens,
                # Tokens may only look backwards
                causal=True,
                # Sliding window size for this layer
                window_size=window_size,
            # Close the call
            )
            # Advance position after last layer processes
            # Only the last layer moves the cache position forward (all layers share one position counter)
            if self.layer_idx == kv_cache.n_layers - 1:
                # Move the cache's write position forward by T tokens
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        # Concatenate the H heads: (B,T,H,D) -> (B,T,C). contiguous() makes memory layout compatible with view()
        y = y.contiguous().view(B, T, -1)
        # Output projection y = y W_o^T (another 2*B*T*C*C FLOPs) mixing information across heads
        y = self.c_proj(y)
        # Return the attention output (to be added to the residual stream)
        return y


# The MLP / feed-forward block: processes each token independently (no mixing across positions)
class MLP(nn.Module):
    # Constructor: create two weight matrices
    def __init__(self, config):
        # Initialize nn.Module base class
        super().__init__()
        # W1: expand C -> 4C ("up projection"). 4*C^2 params: the single biggest matrix in each block
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # W2: contract 4C -> C ("down projection"). Another 4*C^2 params
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    # Forward: x (B,T,C) -> (B,T,C)
    def forward(self, x):
        # h = x W1^T: matmul costing 2*B*T*C*4C FLOPs
        x = self.c_fc(x)
        # Non-linearity relu(h)^2. Without a non-linearity, stacked matmuls would collapse into ONE matrix (no extra power)
        x = F.relu(x).square()
        # out = h W2^T: matmul costing 2*B*T*4C*C FLOPs
        x = self.c_proj(x)
        # Return the MLP output
        return x


# One Transformer block = attention (mix across tokens) + MLP (think per token), each with a residual connection
class Block(nn.Module):
    # Constructor
    def __init__(self, config, layer_idx):
        # Initialize nn.Module base class
        super().__init__()
        # The attention sub-layer
        self.attn = CausalSelfAttention(config, layer_idx)
        # The MLP sub-layer
        self.mlp = MLP(config)

    # Forward pass through one block
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # Residual connection: x = x + Attn(norm(x)). The "+" gives gradients a direct highway backwards (d(x+f)/dx = I + df/dx)
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        # Residual connection around the MLP
        x = x + self.mlp(norm(x))
        # Return the updated residual stream
        return x


# The full GPT language model: tokens in -> probability over the next token out
class GPT(nn.Module):
    # Constructor: build all layers (shapes only; real values come later in init_weights)
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        # Initialize nn.Module base class
        super().__init__()
        # Store the config for later use
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        # Build the list of attention windows, one per layer
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        # Round vocab size up to a multiple of 64: GPU matmuls run fastest when dimensions are multiples of 8/64
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        # If padding actually changed the size...
        if padded_vocab_size != config.vocab_size:
            # ...tell the user
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Container for the main parts of the model
        self.transformer = nn.ModuleDict({
            # Token embedding table: a (V x C) matrix. Row i is the learned vector for token i (a lookup, not a matmul)
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            # The stack of n_layer Transformer blocks
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        # Close the ModuleDict
        })
        # Output "unembedding" matrix (C -> V): turns final token vectors into one score (logit) per vocabulary word
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        # One learnable multiplier per layer for the residual stream
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        # One learnable multiplier per layer for re-injecting the original embedding x0
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        # Tiny (24 -> 1) matrix computing how much of the previous token to mix in
        self.smear_gate = Linear(24, 1, bias=False)
        # Learnable overall strength of the smear
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        # Learnable strength of the "backout" subtraction
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        # Per-head width D
        head_dim = config.n_embd // config.n_head
        # Total width of all KV heads
        kv_dim = config.n_kv_head * head_dim
        # One (V x kv_dim) lookup table per layer that uses value embeddings
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        # Precompute RoPE angles for up to 10x the training context length
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        # Per-head width D (recomputed)
        head_dim = config.n_embd // config.n_head
        # Build the cos/sin tables for RoPE
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # Store cos as a non-trainable buffer (moves with the model to GPU, not updated by the optimizer)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        # Store sin likewise
        self.register_buffer("sin", sin, persistent=False)

    # Disable gradient tracking: initialization isn't something we differentiate through
    @torch.no_grad()
    # Set the starting values of every weight. Good init keeps activations/gradients at sane scales from step 0.
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        # Token embeddings ~ Normal(0, 0.8)
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        # Output matrix ~ Normal(0, 0.001): near-zero so initial predictions are ~uniform over the vocab (loss ~ ln V)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        # Model width C
        n_embd = self.config.n_embd
        # Bound s for Uniform(-s,s) giving std = 1/sqrt(C). Linear algebra: if W has std 1/sqrt(C), then Wx keeps x's scale
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        # Loop over every Transformer block
        for block in self.transformer.h:
            # Init W_q uniformly in (-s, s)
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            # Init W_k
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            # Init W_v
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            # W_o starts at zero: the block initially adds nothing, so the network starts as (nearly) the identity map
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            # W1 of the MLP with a smaller (0.4x) scale
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            # W2 of the MLP starts at zero (same "start as identity" trick)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        # Number of layers
        n_layer = self.config.n_layer
        # For each layer i...
        for i in range(n_layer):
            # ...linearly decay the residual multiplier from 1.15 (first layer) to 1.05 (last layer)
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        # For each layer i...
        for i in range(n_layer):
            # ...linearly decay the x0 blend from 0.20 to 0.05
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized
        # Smear strength starts at 0 (off)
        torch.nn.init.zeros_(self.smear_lambda)
        # Backout strength starts at 0.2
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        # Smear gate weights small and positive
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        # For every value-embedding table...
        for ve in self.value_embeds.values():
            # ...init uniformly with the same scale as W_v
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        # For each block...
        for block in self.transformer.h:
            # ...that has a value-embedding gate...
            if block.attn.ve_gate is not None:
                # ...init its weights small and positive
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        # Per-head width D
        head_dim = self.config.n_embd // self.config.n_head
        # Compute the real RoPE tables (now on a real device, not meta)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # Store them
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        # Unless we are in fp16 mode...
        if COMPUTE_DTYPE != torch.float16:
            # ...store the token embedding table in bf16 to save memory
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            # For each value-embedding table...
            for ve in self.value_embeds.values():
                # ...also store it in bf16
                ve.to(dtype=COMPUTE_DTYPE)

    # Build the RoPE rotation angle tables: angle(t, i) = t * base^(-2i/D)
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        # If no device given...
        if device is None:
            # ...use whatever device the embeddings live on (CPU/GPU)
            device = self.transformer.wte.weight.device
        # stride the channels
        # Even indices 0,2,4,...,D-2: one per 2D feature pair
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # Rotation speed per pair: geometric series from 1 (fast) down to ~1/base (slow). Like clock hands of different speeds
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        # Positions 0..seq_len-1
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        # Outer product: angle matrix of shape (seq_len, D/2), angle[t,i] = t * inv_freq[i]
        freqs = torch.outer(t, inv_freq)
        # Take cos and sin of every angle
        cos, sin = freqs.cos(), freqs.sin()
        # Store in compute dtype (bf16)
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        # Reshape to (1, seq_len, 1, D/2) so they broadcast across batch and heads
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        # Return the tables
        return cos, sin

    # Decide the attention window for every layer
    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        # Uppercase the pattern string
        pattern = config.window_pattern.upper()
        # Validate it only contains S and L
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        # Long window = full context length
        long_window = config.sequence_len
        # Short window = a quarter of context, rounded UP to a multiple of 128 (GPU kernel tile size)
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        # Lookup table from letter to (left, right) window
        char_to_window = {
            # L: look back the full context, 0 tokens forward
            "L": (long_window, 0),
            # S: look back the short window, 0 tokens forward
            "S": (short_window, 0),
        # Close the dict
        }
        # Tile pattern across layers
        # Result list
        window_sizes = []
        # For each layer...
        for layer_idx in range(config.n_layer):
            # ...pick the pattern character, cycling through the pattern
            char = pattern[layer_idx % len(pattern)]
            # ...and append its window size
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        # Override the final layer to full context
        window_sizes[-1] = (long_window, 0)
        # Return the per-layer windows
        return window_sizes

    # Return the device (CPU/GPU) the model lives on
    def get_device(self):
        # Read it from the embedding weight tensor
        return self.transformer.wte.weight.device

    # ***KEY FOR UNDERSTANDING COMPUTE***: how many FLOPs does one training token cost?
    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        # h = heads, q = head dim, t = context length
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        # Running total of attention-score FLOPs (the QK^T and AV matmuls, which have no weights)
        attn_flops = 0
        # For each layer's window...
        for window_size in self.window_sizes:
            # ...take the left (look-back) size
            window = window_size[0]  # (left, right) tuple, we use left
            # Tokens each query actually attends over: full t, or capped by the window
            effective_seq = t if window < 0 else min(window, t)
            # 12 = 2 matmuls (QK^T, AV) x 2 FLOPs (mul+add) x 3 (forward + 2x backward)
            attn_flops += 12 * h * q * effective_seq
        # 6N rule: forward = 2 FLOPs/param, backward = 4 FLOPs/param (grad wrt input AND grad wrt weight)
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        # Return FLOPs per token; total training FLOPs = this x number of training tokens
        return num_flops_per_token

    # Count parameters that take part in matmuls (embeddings are lookups and cost ~0 FLOPs)
    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        # Sum the number of elements in every Linear weight matrix
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        # Return the count
        return matmul_params

    # FLOPs to generate ONE new token at inference with a KV cache (forward only, so 2 per param not 6)
    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        # Number of heads
        h = self.config.n_head
        # Head dim
        q = self.config.n_embd // self.config.n_head
        # Attention FLOPs: 4 = 2 matmuls x 2 FLOPs, over the visible context per layer
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        # Total = weights cost + attention cost
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        # Return it
        return decode_flops

    # FLOPs to process a whole prompt at once (prefill)
    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        # Number of heads
        h = self.config.n_head
        # Head dim
        q = self.config.n_embd // self.config.n_head
        # Running total
        attn_flops = 0
        # For each layer's window
        for window, _ in self.window_sizes:
            # Effective window cannot exceed prompt length
            w = min(window, num_tokens)
            # Sum over tokens of how many tokens each attends to: triangle 1+2+..+w, then flat w for the rest
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            # 4 FLOPs per (head-dim element x attended pair)
            attn_flops += 4 * h * q * attended_tokens
        # Weights cost scales with tokens; add attention cost
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        # Return it
        return prefill_flops

    # Memory needed to cache K and V for one token across all layers
    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        # Head dim
        head_dim = self.config.n_embd // self.config.n_head
        # Bytes per number (2 for bf16)
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        # layers x 2 (K and V) x kv heads x head dim x bytes
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    # Memory that must be READ from the KV cache per decode step (inference is often memory-bandwidth bound)
    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        # Head dim
        head_dim = self.config.n_embd // self.config.n_head
        # Bytes per number
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        # Running total
        total = 0
        # For each layer's window
        for window, _ in self.window_sizes:
            # Add bytes for K and V over the visible tokens
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        # Return total bytes
        return total

    # Count parameters by group (used for scaling-law studies: how loss depends on params and data)
    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        # Token embedding params (V x C)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        # Value embedding params
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        # Output matrix params (C x V)
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        # All block matrices: ~12*C^2 per layer (4C^2 attention + 8C^2 MLP)
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        # All the small scalar params
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        # Grand total
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        # Sanity check that we didn't miss any parameter
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        # Return the breakdown
        return {
            # Embedding count
            'wte': wte,
            # Value embedding count
            'value_embeds': value_embeds,
            # Output head count
            'lm_head': lm_head,
            # Block matrices count
            'transformer_matrices': transformer_matrices,
            # Scalars count
            'scalars': scalars,
            # Total
            'total': total,
        # Close the dict
        }

    # Build the optimizer: different parameter types get different update rules and learning rates
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        # Model width C
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        # 2D weight matrices inside blocks -> optimized with Muon (orthogonalized updates)
        matrix_params = list(self.transformer.h.parameters())
        # Value embedding tables -> AdamW
        value_embeds_params = list(self.value_embeds.parameters())
        # Token embedding table -> AdamW
        embedding_params = list(self.transformer.wte.parameters())
        # Output head -> AdamW
        lm_head_params = list(self.lm_head.parameters())
        # Residual scalars
        resid_params = [self.resid_lambdas]
        # x0 scalars
        x0_params = [self.x0_lambdas]
        # Smear/backout params
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        # Make sure every parameter landed in exactly one group
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        # Wider models need smaller learning rates (otherwise the update to each output becomes too large)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        # Report the scaling factor
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        # List of parameter groups, each with its own hyperparameters
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            # Output head: small LR. betas = decay rates for 1st moment (momentum) and 2nd moment (squared-gradient average)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            # Token embeddings: large LR (each row only gets gradient when its token appears, so updates are sparse)
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            # Value embeddings: half the embedding LR
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            # Residual scalars: very small LR
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            # x0 scalars
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            # Smear params
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        # Close the list
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        # Group matrices by shape so same-shaped ones can be processed together as one big batched tensor
        for shape in sorted({p.shape for p in matrix_params}):
            # All matrices with this shape
            group_params = [p for p in matrix_params if p.shape == shape]
            # Add a Muon group
            param_groups.append(dict(
                # Muon optimizer with its learning rate
                kind='muon', params=group_params, lr=matrix_lr,
                # momentum 0.95; ns_steps = Newton-Schulz iterations used to orthogonalize the update matrix
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            # Close dict and append
            ))

        # Create the combined optimizer
        optimizer = MuonAdamW(param_groups)
        # For each group...
        for group in optimizer.param_groups:
            # ...remember the starting LR so the LR schedule can scale relative to it
            group["initial_lr"] = group["lr"]
        # Return the optimizer
        return optimizer

    # FORWARD PASS: token ids (B,T) -> loss (if targets given) or logits (B,T,V)
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # Batch size and sequence length
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        # Make sure the sequence isn't longer than our precomputed RoPE table
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        # Make sure inputs and tables are on the same device
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        # Make sure tables have the right dtype
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        # Starting position: 0 for training, or how many tokens are already cached during generation
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        # Slice the RoPE tables to positions T0..T0+T
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        # Look up each token id's row in the embedding matrix: (B,T) ints -> (B,T,C) vectors
        x = self.transformer.wte(idx) # embed current token
        # Make sure activations are in compute dtype
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        # Normalize the embeddings
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        # Training path (full sequence available)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            # Need at least 2 tokens to smear
            assert T > 1, "Training forward pass should have T > 1"
            # Gate per position (from position 1 onward) computed from the first 24 channels
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            # Position 0 unchanged; position t gets x_t + gate * x_{t-1}
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        # Inference path with a KV cache
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            # Previous token's embedding from the last call
            x_pre_smear = kv_cache.prev_embedding
            # Save the current last embedding for the next call
            kv_cache.prev_embedding = x[:, -1:, :]
            # Prompt with several tokens
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                # Same gate computation as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                # Same smear as training
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            # Single new token, and we have the previous embedding
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                # Gate for the single token
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                # Mix in previous token's embedding
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        # Keep a copy of the initial embedding
        x0 = x  # save initial normalized embedding for x0 residual
        # Number of layers
        n_layer = self.config.n_layer
        # Layer at which to save the residual for backout
        backout_layer = n_layer // 2  # cache at halfway point
        # Placeholder for the saved residual
        x_backout = None
        # Loop through all blocks
        for i, block in enumerate(self.transformer.h):
            # Rescale the residual stream and blend in the original embedding
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # Look up value embeddings for this layer if it has them
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            # Run the block: attention + MLP (this is where ~all the matmul FLOPs happen)
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            # At the halfway layer...
            if i == backout_layer:
                # ...save the residual stream
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        # If we saved a mid-layer residual...
        if x_backout is not None:
            # ...subtract a learned fraction of it
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        # Final normalization
        x = norm(x)

        # Forward the lm_head (compute logits)
        # Logits will be limited to (-15, 15)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        # Final matmul (B,T,C) x (C,V) -> (B,T,V): a score for every vocab token at every position. 2*B*T*C*V FLOPs
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        # Drop the padded vocab entries
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        # Switch to fp32 for numerically sensitive softmax/log
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        # Smooth cap: 15*tanh(z/15) ~ z for small z, saturates at +-15 (prevents extreme logits)
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # Training mode: we know the correct next tokens
        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            # Cross-entropy loss = -log(softmax(logits)[correct token]), averaged. This scalar is what calculus minimizes.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            # Return the loss (calling loss.backward() later computes dLoss/dW for every weight via the chain rule)
            return loss
        # Inference mode
        else:
            # inference: just return the logits directly
            # Return raw scores
            return logits

    # Disable gradient tracking and autograd bookkeeping for faster generation
    @torch.inference_mode()
    # Simple text generation: repeatedly predict the next token and append it
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        # Input must be a Python list of token ids
        assert isinstance(tokens, list)
        # Device the model is on
        device = self.get_device()
        # Random number generator placeholder
        rng = None
        # If sampling randomly (temperature > 0)...
        if temperature > 0:
            # ...create a generator on the device
            rng = torch.Generator(device=device)
            # ...seeded for reproducibility
            rng.manual_seed(seed)
        # Convert token list to a (1, T) tensor
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        # Generate up to max_tokens new tokens
        for _ in range(max_tokens):
            # Run the whole sequence through the model (naive: recomputes everything each step, no KV cache)
            logits = self.forward(ids) # (B, T, vocab_size)
            # Only the last position's prediction matters for the next token
            logits = logits[:, -1, :] # (B, vocab_size)
            # Optional top-k filtering
            if top_k is not None and top_k > 0:
                # Find the k largest logits
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set everything below the k-th largest to -infinity (probability 0)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Random sampling
            if temperature > 0:
                # Divide by temperature: <1 sharpens, >1 flattens the distribution
                logits = logits / temperature
                # softmax turns scores into probabilities: p_i = exp(z_i) / sum_j exp(z_j)
                probs = F.softmax(logits, dim=-1)
                # Sample one token according to those probabilities
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            # Greedy decoding
            else:
                # Pick the single most likely token
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            # Append the new token to the sequence
            ids = torch.cat((ids, next_ids), dim=1)
            # Convert to a Python int
            token = next_ids.item()
            # Hand the token back to the caller (streaming)
            yield token
