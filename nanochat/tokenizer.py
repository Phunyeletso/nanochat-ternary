"""
BPE Tokenizer in the style of GPT-4: train with rustbpe, inference with tiktoken.
"""
# STUDY NOTE (added): a neural net can only multiply numbers, not letters. The tokenizer turns text into a
# list of integer ids (and back). BPE (Byte Pair Encoding) starts from the 256 possible bytes and repeatedly
# merges the most frequent adjacent pair into a new token, until the vocabulary reaches the target size.
# Each id later selects one row of the embedding matrix (a learned vector of length n_embd).

# Import os for file paths
import os
# Import copy for deep-copying conversations
import copy
# Import an in-memory cache decorator
from functools import lru_cache

# Special control tokens that are not normal text
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    # Start of a document
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    # Start of a user turn
    "<|user_start|>", # user messages
    # End of a user turn
    "<|user_end|>",
    # Start of an assistant turn
    "<|assistant_start|>", # assistant messages
    # End of an assistant turn
    "<|assistant_end|>",
    # Assistant starts writing Python code to run
    "<|python_start|>", # assistant invokes python REPL tool
    # End of the Python code
    "<|python_end|>",
    # Start of the Python output
    "<|output_start|>", # python REPL outputs back to assistant
    # End of the Python output
    "<|output_end|>",
# Close list
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
# Regex that pre-splits text into chunks (words, numbers, punctuation, spaces); merges never cross chunk borders
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
# Import pickle to save/load Python objects
import pickle
# Import the Rust BPE trainer (fast training)
import rustbpe
# Import tiktoken (OpenAI's fast encoder used at inference)
import tiktoken

# The tokenizer class
class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    # Constructor
    def __init__(self, enc, bos_token):
        # The tiktoken encoding object
        self.enc = enc
        # Look up the id of the BOS token
        self.bos_token_id = self.encode_special(bos_token)

    # Alternative constructor
    @classmethod
    # Train a new tokenizer from a stream of text
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        # Create the trainer
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        # Leave room for special tokens
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        # Need at least the 256 base bytes
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        # Run BPE training: learn (vocab_size - 256) merges
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        # Get the regex back
        pattern = tokenizer.get_pattern()
        # Get the learned merges as (bytes, rank) pairs
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        # Convert to dict: token bytes -> id (lower rank = merged earlier)
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        # Special token ids start after the normal tokens
        tokens_offset = len(mergeable_ranks)
        # Assign ids to special tokens
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        # Build the tiktoken encoder
        enc = tiktoken.Encoding(
            # Name
            name="rustbpe",
            # Pre-split regex
            pat_str=pattern,
            # Merges
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            # Special tokens
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        # Close
        )
        # Return a tokenizer instance
        return cls(enc, "<|bos|>")

    # Alternative constructor
    @classmethod
    # Load a saved tokenizer
    def from_directory(cls, tokenizer_dir):
        # Path to the pickle file
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        # Open for binary reading
        with open(pickle_path, "rb") as f:
            # Unpickle the encoder
            enc = pickle.load(f)
        # Return instance
        return cls(enc, "<|bos|>")

    # Alternative constructor
    @classmethod
    # Load one of OpenAI's published tokenizers (e.g. "gpt2", "cl100k_base")
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        # Fetch it
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        # Return instance
        return cls(enc, "<|endoftext|>")

    # Vocabulary size V
    def get_vocab_size(self):
        # Number of token ids
        return self.enc.n_vocab

    # Set of special token strings
    def get_special_tokens(self):
        # From tiktoken
        return self.enc.special_tokens_set

    # Text of a single token id
    def id_to_token(self, id):
        # Decode a one-element list
        return self.enc.decode([id])

    # Cache results (special-token lookups are repeated a lot)
    @lru_cache(maxsize=32)
    # Id of a special token string
    def encode_special(self, text):
        # Look it up
        return self.enc.encode_single_token(text)

    # Id of BOS
    def get_bos_token_id(self):
        # Stored at init
        return self.bos_token_id

    # Text -> token ids
    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        # If something to prepend...
        if prepend is not None:
            # ...resolve it to an id
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        # If something to append...
        if append is not None:
            # ...resolve it to an id
            append_id = append if isinstance(append, int) else self.encode_special(append)

        # Single string
        if isinstance(text, str):
            # Encode (treating special-token text as normal text, for safety)
            ids = self.enc.encode_ordinary(text)
            # Prepend if asked
            if prepend is not None:
                # Insert at the start
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            # Append if asked
            if append is not None:
                # Add at the end
                ids.append(append_id)
        # List of strings
        elif isinstance(text, list):
            # Encode all in parallel threads
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            # Prepend if asked
            if prepend is not None:
                # For each row
                for ids_row in ids:
                    # Insert at start
                    ids_row.insert(0, prepend_id) # TODO: same
            # Append if asked
            if append is not None:
                # For each row
                for ids_row in ids:
                    # Add at end
                    ids_row.append(append_id)
        # Anything else
        else:
            # Error
            raise ValueError(f"Invalid input type: {type(text)}")

        # Return ids
        return ids

    # Calling the tokenizer like a function = encode
    def __call__(self, *args, **kwargs):
        # Delegate
        return self.encode(*args, **kwargs)

    # Token ids -> text
    def decode(self, ids):
        # Delegate to tiktoken
        return self.enc.decode(ids)

    # Raw bytes of a single token
    def decode_single_token_bytes(self, token_id):
        # Delegate
        return self.enc.decode_single_token_bytes(token_id)

    # Save to disk
    def save(self, tokenizer_dir):
        # save the encoding object to disk
        # Make the folder
        os.makedirs(tokenizer_dir, exist_ok=True)
        # File path
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        # Open for binary writing
        with open(pickle_path, "wb") as f:
            # Pickle the encoder
            pickle.dump(self.enc, f)
        # Report
        print(f"Saved tokenizer encoding to {pickle_path}")

    # Turn a chat conversation into token ids + a training mask
    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        # Output lists
        ids, mask = [], []
        # Helper to append tokens with a mask value
        def add_tokens(token_ids, mask_val):
            # Wrap a single id in a list
            if isinstance(token_ids, int):
                # Make it a list
                token_ids = [token_ids]
            # Append ids
            ids.extend(token_ids)
            # Append matching mask values (1 = model is trained to predict this token, 0 = context only)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        # If there's a system message...
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            # Copy so we don't modify the caller's data
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            # Messages list
            messages = conversation["messages"]
            # Next must be a user message
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            # Prepend system text to the user message
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            # Drop the system message
            messages = messages[1:]
        # No system message
        else:
            # Use as is
            messages = conversation["messages"]
        # At least one message
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        # BOS id
        bos = self.get_bos_token_id()
        # User delimiters
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        # Assistant delimiters
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        # Python delimiters
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        # Output delimiters
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        # Start with BOS (not trained on)
        add_tokens(bos, 0)
        # For each message
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            # Messages must alternate user/assistant
            must_be_from = "user" if i % 2 == 0 else "assistant"
            # Check
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            # Message content
            content = message["content"]

            # User message
            if message["role"] == "user":
                # Must be a plain string
                assert isinstance(content, str), "User messages are simply expected to be strings"
                # Encode it
                value_ids = self.encode(content)
                # User start (mask 0: we don't train the model to write user text)
                add_tokens(user_start, 0)
                # User text
                add_tokens(value_ids, 0)
                # User end
                add_tokens(user_end, 0)
            # Assistant message
            elif message["role"] == "assistant":
                # Assistant start (mask 0: it's given as a prompt)
                add_tokens(assistant_start, 0)
                # Plain text
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    # Encode
                    value_ids = self.encode(content)
                    # Mask 1: train the model to produce these tokens
                    add_tokens(value_ids, 1)
                # List of parts
                elif isinstance(content, list):
                    # For each part
                    for part in content:
                        # Encode its text
                        value_ids = self.encode(part["text"])
                        # Normal text part
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            # Train on it
                            add_tokens(value_ids, 1)
                        # Python code part
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            # Train on start marker
                            add_tokens(python_start, 1)
                            # Train on code
                            add_tokens(value_ids, 1)
                            # Train on end marker
                            add_tokens(python_end, 1)
                        # Python output part
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            # Not trained
                            add_tokens(output_start, 0)
                            # Not trained
                            add_tokens(value_ids, 0)
                            # Not trained
                            add_tokens(output_end, 0)
                        # Unknown part type
                        else:
                            # Error
                            raise ValueError(f"Unknown part type: {part['type']}")
                # Unknown content type
                else:
                    # Error
                    raise ValueError(f"Unknown content type: {type(content)}")
                # Train the model to end its turn
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        # Truncate ids
        ids = ids[:max_tokens]
        # Truncate mask
        mask = mask[:max_tokens]
        # Return both
        return ids, mask

    # Debug helper: colour tokens by mask
    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        # Terminal colour code red
        RED = '\033[91m'
        # Green
        GREEN = '\033[92m'
        # Reset
        RESET = '\033[0m'
        # Gray
        GRAY = '\033[90m'
        # Output pieces
        tokens = []
        # For each token and mask value
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            # Decode the token
            token_str = self.decode([token_id])
            # Green if trained on, red otherwise
            color = GREEN if mask_val == 1 else RED
            # Add coloured text
            tokens.append(f"{color}{token_str}{RESET}")
            # Optionally show the id
            if with_token_id:
                # Add id
                tokens.append(f"{GRAY}({token_id}){RESET}")
        # Join with separators
        return '|'.join(tokens)

    # Render a conversation so the model will write the next assistant reply
    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        # Copy
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        # Messages
        messages = conversation["messages"]
        # Last must be assistant
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        # Remove it (the model will generate it)
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        # Tokenize the rest
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        # Assistant start id
        assistant_start = self.encode_special("<|assistant_start|>")
        # Append it so the model starts answering
        ids.append(assistant_start)
        # Return ids
        return ids

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

# Load the project's tokenizer
def get_tokenizer():
    # Import base dir helper
    from nanochat.common import get_base_dir
    # Base dir
    base_dir = get_base_dir()
    # Tokenizer folder
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # Load and return
    return RustBPETokenizer.from_directory(tokenizer_dir)

# Load the bytes-per-token table
def get_token_bytes(device="cpu"):
    # Import torch
    import torch
    # Import base dir helper
    from nanochat.common import get_base_dir
    # Base dir
    base_dir = get_base_dir()
    # Tokenizer folder
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # File path
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    # Must exist
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    # Open
    with open(token_bytes_path, "rb") as f:
        # Load tensor onto the device
        token_bytes = torch.load(f, map_location=device)
    # Return it
    return token_bytes
