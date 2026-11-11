"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use DAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""
# STUDY NOTE (added): the calculus of RL (policy gradient / REINFORCE):
#   We want to maximize expected reward J = E[R]. The gradient trick:  dJ/dW = E[ (R - baseline) * d log p(sample)/dW ].
#   So: sample answers, score them (correct = 1, wrong = 0), then push UP the log-probability of tokens in
#   above-average answers and DOWN for below-average ones. Same backward() machinery, just a different loss.
#   Compute cost: generating samples (inference) dominates; each step generates 16 answers x 16 questions.

# Import argparse
import argparse
# Import os
import os
# Import itertools (for cycling through data)
import itertools
# Import wandb
import wandb
# Import PyTorch
import torch
# Import distributed communication
import torch.distributed as dist
# Import setup helpers
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
# Import checkpoint helpers
from nanochat.checkpoint_manager import save_checkpoint, load_model
# Import inference engine (to sample answers)
from nanochat.engine import Engine
# Import math word-problem dataset (has a reward function: is the final answer correct?)
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
# Parser
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
# Run name
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
# Device
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
# Model tag
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
# Model step
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
# Passes over GSM8K
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
# Batch sizes / sampling
# Sequences per forward pass
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
# Questions per optimizer step
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
# Answers sampled per question
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
# Generation
# Max answer length
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
# Randomness of sampling
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
# Only sample from the 50 most likely tokens
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
# Embedding LR
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
# Output head LR
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
# Matrix LR
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
# Weight decay
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
# Start at 5% of base LR (RL needs gentle updates)
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
# Eval frequency
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
# Eval size
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
# Save frequency
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
# Parse
args = parser.parse_args()
# Copy for logging
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
# Device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# Init runtime
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# GPU 0?
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

# wandb logging init
# Dummy?
use_dummy_wandb = args.run == "dummy" or not master_process
# Logger
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
# Start from the SFT (chat) model
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
# Engine for generating samples
engine = Engine(model, tokenizer) # for sampling rollouts

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

# Training questions
train_task = GSM8K(subset="main", split="train")
# Test questions
val_task = GSM8K(subset="main", split="test")
# Total optimizer steps
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
# Report
print0(f"Calculated number of steps: {num_steps}")

# Sampling needs no gradients
@torch.no_grad()
# Generator: for each question, sample answers, score them, and build training tensors
def get_batch():
    # Token used for padding
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    # This GPU's question indices
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    # Loop over them forever
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        # The question + reference answer
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        # Prompt tokens
        tokens = tokenizer.render_for_completion(conversation)
        # Prompt length
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        # Eval mode for sampling
        model.eval() # ensure the model is in eval mode
        # Sampled token sequences
        generated_token_sequences = []
        # Masks (1 = model-generated token)
        masks = []
        # Number of sampling rounds
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
        # For each round
        for sampling_step in range(num_sampling_steps):
            # Different random seed each round
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            # Sample a batch of answers
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                # Prompt
                tokens,
                # How many
                num_samples=args.device_batch_size,
                # Max length
                max_tokens=args.max_new_tokens,
                # Temperature
                temperature=args.temperature,
                # Top-k
                top_k=args.top_k,
                # Seed
                seed=seed, # must make sure to change the seed for each sampling step
            # Close
            )
            # Collect sequences
            generated_token_sequences.extend(generated_token_sequences_batch)
            # Collect masks
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        # Rewards list
        rewards = []
        # For each sample
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            # Answer tokens
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            # Answer text
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            # 1 if the final number is correct, else 0
            reward = train_task.reward(conversation, generated_text)
            # Record
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        # Longest sequence
        max_length = max(len(seq) for seq in generated_token_sequences)
        # Pad sequences
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        # Pad masks with 0
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        # (N, L) token ids
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        # (N, L) mask
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        # Inputs
        inputs = ids[:, :-1]
        # Targets (copy)
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        # Ignore prompt/padding/tool-output targets
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        # Rewards tensor
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        # Average reward (the baseline)
        mu = rewards.mean()
        # Advantage = how much better than average each sample was (positive = reinforce, negative = discourage)
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        # Yield everything
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
# Evaluate: for each question, sample k answers and record which are correct
def run_gsm8k_eval(task, tokenizer, engine,
    # Max questions
    max_examples=None,
    # Answers per question
    num_samples=1,
    # Max answer length
    max_completion_tokens=256,
    # Temperature
    temperature=0.0,
    # Top-k
    top_k=50
# Close signature
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    # Number of questions
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    # This GPU's questions
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        # Question
        conversation = task[idx]
        # Prompt tokens
        tokens = tokenizer.render_for_completion(conversation)
        # Prompt length
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        # Must fit in one batch
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
        # Sample answers
        generated_token_sequences, masks = engine.generate_batch(
            # Prompt
            tokens,
            # How many
            num_samples=num_samples,
            # Max length
            max_tokens=max_completion_tokens,
            # Temperature
            temperature=temperature,
            # Top-k
            top_k=top_k
        # Close
        )
        # Check each sample for correctness
        # Outcomes
        outcomes = []
        # For each sample
        for sample_tokens in generated_token_sequences:
            # Answer tokens
            generated_tokens = sample_tokens[prefix_length:]
            # Answer text
            generated_text = tokenizer.decode(generated_tokens)
            # Correct?
            is_correct = task.evaluate(conversation, generated_text)
            # Record
            outcomes.append({
                # Flag
                "is_correct": is_correct
            # Close
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        # Record for this question
        record = {
            # Index
            "idx": idx,
            # Outcomes
            "outcomes": outcomes,
        # Close
        }
        # Yield
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
# Build optimizer
optimizer = model.setup_optimizer(
    # Output head LR
    unembedding_lr=args.unembedding_lr,
    # Embedding LR
    embedding_lr=args.embedding_lr,
    # Matrix LR
    matrix_lr=args.matrix_lr,
    # Weight decay
    weight_decay=args.weight_decay,
# Close
)

# Set the initial learning rate as a fraction of the base learning rate
# For each group
for group in optimizer.param_groups:
    # Scale down
    group["lr"] = group["lr"] * args.init_lr_frac
    # Remember
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
# LR multiplier
def get_lr_multiplier(it):
    # Linear from 1 to 0
    lrm = 1.0 - it / num_steps
    # Return
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
# Report total sequences
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
# Must divide evenly
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
# Questions per GPU per step
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
# Report
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
# Rollout generator
batch_iterator = get_batch()
# For each step
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    # Time to evaluate?
    if step % args.eval_every == 0:
        # Eval mode
        model.eval()
        # pass@k counters
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
        # Evaluation iterator
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        # Run it
        records = list(records_iter) # collect all records
        # For k = 1..B
        for k in range(1, args.device_batch_size + 1):
            # Count questions where at least one of the first k samples is correct
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        # Number of questions
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        # Multi-GPU
        if ddp:
            # Sum counts
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            # Sum pass counts
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        # Convert to fractions
        passk = passk / num_records.item() # normalize by the total number of records
        # Format
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        # Report
        print0(f"Step {step} | {', '.join(print_passk)}")
        # For logging
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        # Log
        wandb_run.log({
            # Step
            "step": step,
            # pass@k values
            **log_passk,
        # Close
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    # Rewards for logging
    rewards_list = []
    # Lengths for logging
    sequence_lengths = []
    # For each question this GPU handles
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        # Sample and score answers
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # Evaluate the loss and gradients
        # Train mode
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        # Must divide evenly
        assert inputs_all.size(0) % args.device_batch_size == 0
        # Number of forward passes
        num_passes = inputs_all.size(0) // args.device_batch_size
        # For each pass
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            # Slice bounds
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            # Inputs
            inputs = inputs_all[b0:b1]
            # Targets
            targets = targets_all[b0:b1]
            # Rewards
            rewards = rewards_all[b0:b1]
            # Advantages
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            # log p(token) for every generated token, shape (B, T)
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            # Policy-gradient objective: sum of log p(token) weighted by its sample's advantage
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            # Count of trained tokens
            num_valid = (targets >= 0).sum().clamp(min=1)
            # Normalize to a per-token average
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            # Minimizing -objective = maximizing objective (gradient ASCENT on reward)
            loss = -pg_obj
            # Backprop: accumulate gradients
            loss.backward()
            # Report
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        # Mean reward
        rewards_list.append(rewards_all.mean().item())
        # Lengths
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    # Mean reward
    mean_reward = sum(rewards_list) / len(rewards_list)
    # Mean length
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    # Multi-GPU
    if ddp: # aggregate across ranks
        # Reward tensor
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        # Length tensor
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        # Average across GPUs
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        # Average across GPUs
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        # Back to float
        mean_reward = mean_reward_tensor.item()
        # Back to float
        mean_sequence_length = mean_sequence_length_tensor.item()
    # Report
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    # Log
    wandb_run.log({
        # Step
        "step": step,
        # Reward
        "reward": mean_reward,
        # Length
        "sequence_length": mean_sequence_length,
    # Close
    })

    # Update the model parameters
    # LR multiplier
    lrm = get_lr_multiplier(step)
    # For each group
    for group in optimizer.param_groups:
        # Set LR
        group["lr"] = group["initial_lr"] * lrm
    # Apply update
    optimizer.step()
    # Clear gradients
    model.zero_grad(set_to_none=True)
    # Log
    wandb_run.log({
        # Step
        "step": step,
        # LR mult
        "lrm": lrm,
    # Close
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    # Time to save?
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        # Base dir
        base_dir = get_base_dir()
        # Depth
        depth = model.config.n_layer
        # Folder name
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # base the model tag on the depth of the base model
        # Path
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        # Config as dict
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        # Save
        save_checkpoint(
            # Folder
            checkpoint_dir,
            # Step
            step,
            # Weights
            model.state_dict(),
            # No optimizer
            None, # note: we don't bother to save the optimizer state
            # Metadata
            {
                # Config
                "model_config": model_config_kwargs,
            # Close
            }
        # Close
        )
        # Report
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Close wandb
wandb_run.finish() # wandb run finish
# Tear down distributed
compute_cleanup()
