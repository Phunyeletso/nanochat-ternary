"""
Training script for the ternary experiments. A lean, single-process copy of scripts/base_train.py
(same data pipeline, LR/momentum/weight-decay schedules, batch-size LR scaling, val bpb eval),
with ternary options added behind flags. scripts/base_train.py itself is untouched.

Full precision (should reproduce base_train):
    python -m ternary.train --tag fp
Phase 1 ternary layers (ternary weights frozen at init; only non-ternary parts train):
    python -m ternary.train --tag t1 --ternary-linear 1 --flip-rule none

Log format matches base_train ("step NNNNN/NNNNN ... | loss: x", "Step NNNNN | Validation bpb: x")
so ternary/plot_log.py works on both. Per-step metrics also go to <tag>.jsonl.
"""
import argparse
import gc
import json
import math
import os
import time

import torch

from nanochat.common import compute_init, print0, get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import (tokenizing_distributed_data_loader_with_state_bos_bestfit,
                                 tokenizing_distributed_data_loader_bos_bestfit)
from nanochat.loss_eval import evaluate_bpb
from ternary.flip import make_flip_rule
from ternary.layers import qmax_for_tables
from ternary.bitnet import convert_to_bitnet
from ternary.model import (TernaryConfig, convert_to_ternary, setup_optimizer_ternary,
                           ternary_modules, set_capture_grad, param_report)

p = argparse.ArgumentParser()
p.add_argument("--tag", type=str, required=True)
p.add_argument("--out-dir", type=str, default="/home/claude/runs")
# model (defaults = Phase 0 baseline)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--aspect-ratio", type=int, default=64)
p.add_argument("--head-dim", type=int, default=64)
p.add_argument("--max-seq-len", type=int, default=256)
p.add_argument("--window-pattern", type=str, default="L")
# ternary flags (all off = original model)
p.add_argument("--ternary-linear", type=int, default=0, help="1 = make c_q,c_k,c_v,attn.c_proj,mlp.c_fc,mlp.c_proj ternary")
p.add_argument("--ternary-embed", type=int, default=0, help="1 = token embedding table ternary")
p.add_argument("--ternary-value-embeds", type=int, default=0, help="1 = value-embedding tables ternary")
p.add_argument("--ternary-lm-head", type=int, default=0, help="1 = final output layer ternary")
p.add_argument("--ternary-gates", type=int, default=0, help="1 = the tiny ve_gate/smear_gate matrices ternary")
p.add_argument("--ternary-all", type=int, default=0, help="1 = everything ternary: block matrices, token table, value tables, output layer, gates")
p.add_argument("--act-tables", type=int, default=0, help="0 = 8-bit data before ternary layers; k = round to what k balanced-ternary tables hold (+-(3^k-1)/2)")
p.add_argument("--quant-act", type=int, default=1, help="1 = 8-bit absmax activations into ternary linears")
p.add_argument("--flip-rule", type=str, default="none", choices=["none", "topk", "stochastic", "counter"],
               help="how ternary weights are trained (none = frozen). 'counter' is the fallback that keeps hidden per-weight state")
p.add_argument("--flip-frac", type=float, default=1e-3, help="topk: fraction of each matrix flipped per step at step 0 (decays linearly to 0)")
p.add_argument("--flip-lr", type=float, default=1e-3, help="stochastic: flip probability scale (decays linearly to 0)")
p.add_argument("--flip-schedule", type=str, default="linear", choices=["linear", "cosine", "constant"],
               help="how the flip rate changes over training (linear to --flip-floor = Phase 2/3 default)")
p.add_argument("--flip-floor", type=float, default=0.0, help="fraction of the starting flip rate kept at the end")
p.add_argument("--counter-threshold", type=int, default=8, help="counter fallback: flips after this many net votes")
p.add_argument("--seed", type=int, default=0, help="changes weight init (torch seed 42+seed) and the stochastic flip RNG; 0 = nanochat's default init")
p.add_argument("--bitnet", type=int, default=0, help="1 = baseline B: standard BitNet b1.58 (latent weights + STE) on the six block matrices")
p.add_argument("--ternary-scale-lr", type=float, default=0.002)
# optimisation (defaults = base_train)
p.add_argument("--num-iterations", type=int, default=600)
p.add_argument("--device-batch-size", type=int, default=32)
p.add_argument("--total-batch-size", type=int, default=8192)
p.add_argument("--embedding-lr", type=float, default=0.3)
p.add_argument("--unembedding-lr", type=float, default=0.008)
p.add_argument("--weight-decay", type=float, default=0.28)
p.add_argument("--matrix-lr", type=float, default=0.02)
p.add_argument("--scalar-lr", type=float, default=0.5)
p.add_argument("--warmup-steps", type=int, default=40)
p.add_argument("--warmdown-ratio", type=float, default=0.65)
p.add_argument("--final-lr-frac", type=float, default=0.05)
p.add_argument("--target-param-data-ratio", type=float, default=12)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--eval-tokens", type=int, default=131072)
p.add_argument("--compile", type=int, default=0)
p.add_argument("--ckpt-every", type=int, default=0, help="save a resumable checkpoint every N steps (0 = off)")
p.add_argument("--resume", type=int, default=0, help="1 = continue from <out-dir>/<tag>.resume.pt if it exists")
p.add_argument("--save", type=int, default=1, help="1 = save the trained model to <out-dir>/<tag>.pt (for ternary/generate.py)")
args = p.parse_args()

ddp, rank, local_rank, world, device = compute_init("cpu")
os.makedirs(args.out_dir, exist_ok=True)
RESUME_PATH = os.path.join(args.out_dir, f"{args.tag}.resume.pt")
resume_ck = torch.load(RESUME_PATH, map_location="cpu", weights_only=False) if (args.resume and os.path.exists(RESUME_PATH)) else None
jsonl = open(os.path.join(args.out_dir, f"{args.tag}.jsonl"), "a" if resume_ck else "w")
print0(f"args: {json.dumps(vars(args))}")

tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()


def build_model_meta(depth):
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    cfg = GPTConfig(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=depth,
                    n_head=model_dim // args.head_dim, n_kv_head=model_dim // args.head_dim,
                    n_embd=model_dim, window_pattern=args.window_pattern)
    with torch.device("meta"):
        return GPT(cfg)


model = build_model_meta(args.depth)
model.to_empty(device=device)
torch.manual_seed(42 + args.seed)  # 42 is what compute_init uses, so seed 0 = original init
model.init_weights()

# scaling rules exactly as in base_train (computed on the float model, before conversion)
def scaling_params(m):
    c = m.num_scaling_params()
    return c["transformer_matrices"] + c["lm_head"]
target_tokens = int(args.target_param_data_ratio * scaling_params(model))
D_REF = args.target_param_data_ratio * scaling_params(build_model_meta(12))
B_REF = 2 ** 19
batch_lr_scale = (args.total_batch_size / B_REF) ** 0.5
weight_decay_scaled = args.weight_decay * math.sqrt(args.total_batch_size / B_REF) * (D_REF / target_tokens)

A_ = bool(args.ternary_all)
tcfg = TernaryConfig(linear=bool(args.ternary_linear) or A_, embed=bool(args.ternary_embed) or A_,
                     value_embeds=bool(args.ternary_value_embeds) or A_, quant_act=bool(args.quant_act),
                     lm_head=bool(args.ternary_lm_head) or A_, gates=bool(args.ternary_gates) or A_,
                     act_qmax=qmax_for_tables(args.act_tables))
is_ternary = tcfg.linear or tcfg.embed or tcfg.value_embeds or tcfg.lm_head or tcfg.gates
assert not (is_ternary and args.bitnet), "--bitnet is a separate baseline; do not combine with --ternary-*"
if args.bitnet:
    convert_to_bitnet(model)
    print0("baseline B: BitNet b1.58 latent-weight layers on c_q,c_k,c_v,attn.c_proj,mlp.c_fc,mlp.c_proj")
if is_ternary:
    convert_to_ternary(model, tcfg)
    set_capture_grad(model, args.flip_rule != "none")  # frozen weights -> no need to compute dL/dW
    optimizer = setup_optimizer_ternary(model, unembedding_lr=args.unembedding_lr * batch_lr_scale,
                                        embedding_lr=args.embedding_lr * batch_lr_scale,
                                        scalar_lr=args.scalar_lr * batch_lr_scale,
                                        matrix_lr=args.matrix_lr * batch_lr_scale,
                                        weight_decay=weight_decay_scaled,
                                        ternary_scale_lr=args.ternary_scale_lr)
else:
    optimizer = model.setup_optimizer(unembedding_lr=args.unembedding_lr * batch_lr_scale,
                                      embedding_lr=args.embedding_lr * batch_lr_scale,
                                      scalar_lr=args.scalar_lr * batch_lr_scale,
                                      matrix_lr=args.matrix_lr * batch_lr_scale,
                                      weight_decay=weight_decay_scaled)
print0(f"ternary config: {tcfg}")
print0(f"storage: {json.dumps(param_report(model))}")
if is_ternary:
    from ternary.model import decimal_param_report
    dec = decimal_param_report(model)
    print0(f"remaining decimal parameters: {sum(dec.values()):,} values in {len(dec)} tensors")
print0(f"batch_lr_scale={batch_lr_scale:.4f} weight_decay_scaled={weight_decay_scaled:.6f}")

flipper = make_flip_rule(args.flip_rule, args.num_iterations, args.flip_frac, args.flip_lr, args.counter_threshold,
                         args.flip_schedule, args.flip_floor) if is_ternary else None
if flipper is not None and flipper.breaks_no_hidden_copy_rule:
    print0("WARNING: flip rule 'counter' keeps an int8 counter per weight (breaks the no-hidden-copy rule)")
flip_gen = torch.Generator(device=device).manual_seed(args.seed)
train_model = torch.compile(model, dynamic=False) if args.compile else model

if resume_ck:
    model.load_state_dict(resume_ck["model"])
    optimizer.load_state_dict(resume_ck["optimizer"])
    flip_gen.set_state(resume_ck["flip_gen"])
    torch.set_rng_state(resume_ck["torch_rng"])
    print0(f"RESUMED from step {resume_ck['step']} ({RESUME_PATH})")
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
                                                                         resume_state_dict=resume_ck["loader_state"] if resume_ck else None)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, loader_state = next(train_loader)

N = args.num_iterations
def get_lr_multiplier(it):
    wd_iters = round(args.warmdown_ratio * N)
    if it < args.warmup_steps:
        return (it + 1) / args.warmup_steps
    elif it <= N - wd_iters:
        return 1.0
    progress = (N - it) / wd_iters
    return progress + (1 - progress) * args.final_lr_frac

def get_muon_momentum(it):
    wd_iters = round(args.warmdown_ratio * N)
    start = N - wd_iters
    if it < 400:
        f = it / 400
        return (1 - f) * 0.85 + f * 0.97
    elif it >= start:
        pr = (it - start) / wd_iters
        return 0.97 * (1 - pr) + 0.90 * pr
    return 0.97

def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / N))

def zero_fraction(m):
    mods = ternary_modules(m)
    if not mods:
        return None
    z = sum((mm.weight_t == 0).sum().item() for _, mm in mods)
    n = sum(mm.weight_t.numel() for _, mm in mods)
    return z / n

grad_accum = args.total_batch_size // (args.device_batch_size * args.max_seq_len)
assert grad_accum * args.device_batch_size * args.max_seq_len == args.total_batch_size
smooth, total_time, min_bpb, step = 0.0, 0.0, float("inf"), 0
if resume_ck:
    smooth, total_time, min_bpb, step = resume_ck["smooth"], resume_ck["total_time"], resume_ck["min_bpb"], resume_ck["step"]


def save_resume():
    tmp = RESUME_PATH + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "flip_gen": flip_gen.get_state(),
                "torch_rng": torch.get_rng_state(), "loader_state": loader_state, "step": step, "smooth": smooth,
                "total_time": total_time, "min_bpb": min_bpb}, tmp)
    os.replace(tmp, RESUME_PATH)   # atomic: a restart mid-save never leaves a broken checkpoint
while True:
    last = step == N
    if args.eval_every > 0 and (last or step % args.eval_every == 0):
        model.eval()
        steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len)
        with torch.no_grad():
            bpb = evaluate_bpb(model, build_val_loader(), steps, token_bytes)
        min_bpb = min(min_bpb, bpb)
        print0(f"Step {step:05d} | Validation bpb: {bpb:.6f}")
        jsonl.write(json.dumps({"step": step, "val_bpb": bpb}) + "\n"); jsonl.flush()
        model.train()
    if last:
        break

    t0 = time.time()
    for _ in range(grad_accum):
        loss = train_model(x, y)
        raw = loss.detach()
        (loss / grad_accum).backward()
        x, y, loader_state = next(train_loader)
    lrm = get_lr_multiplier(step)
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * lrm
        if g["kind"] == "muon":
            g["momentum"] = get_muon_momentum(step)
            g["weight_decay"] = get_weight_decay(step)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    if flipper is not None:
        flip_frac = flipper.step(model, step, flip_gen)   # reads dL/dW, flips, discards dL/dW
    else:
        flip_frac = 0.0
        for _, m in ternary_modules(model):
            m.clear_weight_grad()
    raw_f = raw.item()
    dt = time.time() - t0
    if step > 10:
        total_time += dt
    smooth = 0.9 * smooth + 0.1 * raw_f
    dsmooth = smooth / (1 - 0.9 ** (step + 1))
    zf = zero_fraction(model)
    extra = f" | flipped: {flip_frac:.6f} | zeros: {zf:.4f}" if zf is not None else ""
    print0(f"step {step:05d}/{N:05d} | loss: {dsmooth:.6f} | raw: {raw_f:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/s: {int(args.total_batch_size/dt):,}{extra}")
    jsonl.write(json.dumps({"step": step, "loss_smooth": dsmooth, "loss": raw_f, "dt": dt,
                            "flip_frac": flip_frac if zf is not None else None, "zero_frac": zf}) + "\n")
    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    step += 1
    if args.ckpt_every and step % args.ckpt_every == 0 and step < N:
        save_resume()

if args.save:
    from dataclasses import asdict
    ckpt = os.path.join(args.out_dir, f"{args.tag}.pt")
    torch.save({"args": vars(args), "gpt_config": asdict(model.config), "state_dict": model.state_dict()}, ckpt)
    print0(f"Saved model to {ckpt} ({os.path.getsize(ckpt)/1e6:.1f} MB)")
print0(f"Total training time: {total_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_bpb:.6f}")
