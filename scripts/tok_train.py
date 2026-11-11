"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
# STUDY NOTE (added): step 1 of the pipeline. No neural network or calculus here: BPE is a greedy counting
# algorithm. Vocab size V matters for compute later: the embedding table and output head are V x C matrices,
# and the final matmul costs 2*C*V FLOPs per token.

# Import os
import os
# Import time
import time
# Import argparse
import argparse
# Import PyTorch (to save the token_bytes tensor)
import torch
# Import the tokenizer class
from nanochat.tokenizer import RustBPETokenizer
# Import base dir helper
from nanochat.common import get_base_dir
# Import the document iterator
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

# Parser
parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
# How much text to learn from
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
# Truncate long documents
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
# Target vocab size
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
# Parse
args = parser.parse_args()
# Report
print(f"max_chars: {args.max_chars:,}")
# Report
print(f"doc_cap: {args.doc_cap:,}")
# Report
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

# Generator of training documents
def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    # Characters seen so far
    nchars = 0
    # For each batch of docs
    for batch in parquets_iter_batched(split="train"):
        # For each doc
        for doc in batch:
            # Its text
            doc_text = doc
            # Too long?
            if len(doc_text) > args.doc_cap:
                # Truncate
                doc_text = doc_text[:args.doc_cap]
            # Count characters
            nchars += len(doc_text)
            # Hand to the trainer
            yield doc_text
            # Enough text?
            if nchars > args.max_chars:
                # Stop
                return
# Create the iterator
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
# Start timer
t0 = time.time()
# Run BPE training
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
# Stop timer
t1 = time.time()
# Duration
train_time = t1 - t0
# Report
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
# Base dir
base_dir = get_base_dir()
# Tokenizer folder
tokenizer_dir = os.path.join(base_dir, "tokenizer")
# Save
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
# Test string with numbers, contractions, symbols, unicode
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
# Text -> ids
encoded = tokenizer.encode(test_text)
# ids -> text
decoded = tokenizer.decode(encoded)
# Round trip must be lossless
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
# Vocab size
vocab_size = tokenizer.get_vocab_size()
# Ids of special tokens
special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
# Bytes per token list
token_bytes = []
# For every token id
for token_id in range(vocab_size):
    # Special token?
    if token_id in special_ids:
        # Count as 0 bytes
        token_bytes.append(0) # special tokens are not counted
    # Normal token
    else:
        # use the raw bytes of the token: decoding to a string first corrupts
        # tokens that are not valid standalone UTF-8 (e.g. the raw bytes >= 0x80)
        # Number of raw bytes
        num_bytes = len(tokenizer.decode_single_token_bytes(token_id))
        # Record
        token_bytes.append(num_bytes)
# Convert to tensor
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
# File path
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
# Open
with open(token_bytes_path, "wb") as f:
    # Save
    torch.save(token_bytes, f)
# Report
print(f"Saved token_bytes to {token_bytes_path}")
