"""
Turn an initialised nanochat GPT into a ternary model, behind flags. nanochat/gpt.py is untouched.

    model = GPT(config); model.to_empty(device); model.init_weights()   # normal nanochat
    convert_to_ternary(model, TernaryConfig(...))                       # this file
    optimizer = setup_optimizer_ternary(model, ...)

What gets converted (TernaryConfig defaults):
  * linear=True        -> attn.c_q, attn.c_k, attn.c_v, attn.c_proj, mlp.c_fc, mlp.c_proj  (every block)
  * embed=False        -> transformer.wte (token embedding table)
  * value_embeds=False -> value_embeds[*] (nanochat's second vocab-sized table)
Kept in normal precision: lm_head, norms (parameter-free), ve_gate, smear_gate, per-layer scalars.
"""
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from nanochat.optim import MuonAdamW
from nanochat.common import print0
from ternary.layers import TernaryLinear, TernaryEmbedding, _TernaryBase

TARGETS = [("attn", "c_q"), ("attn", "c_k"), ("attn", "c_v"), ("attn", "c_proj"), ("mlp", "c_fc"), ("mlp", "c_proj")]


@dataclass
class TernaryConfig:
    linear: bool = True          # the six block matrices
    embed: bool = False          # token embedding table
    value_embeds: bool = False   # value-embedding tables
    quant_act: bool = True       # 8-bit absmax activations into TernaryLinear
    lm_head: bool = False        # final output layer (Phase 6)
    gates: bool = False          # the tiny ve_gate and smear_gate matrices (Phase 6)
    act_qmax: int = 127          # data rounding before ternary layers: 127 = 8-bit; (3^k-1)/2 = k coin tables (Phase 7)


def convert_to_ternary(model, cfg: TernaryConfig):
    if cfg.linear:
        for block in model.transformer.h:
            for sub, name in TARGETS:
                parent = getattr(block, sub)
                lin = getattr(parent, name)
                setattr(parent, name, TernaryLinear.from_float(lin.weight, quant_act=cfg.quant_act, act_qmax=cfg.act_qmax))
    if cfg.embed:
        model.transformer.wte = TernaryEmbedding.from_float(model.transformer.wte.weight)
    if cfg.value_embeds:
        for k in list(model.value_embeds.keys()):
            model.value_embeds[k] = TernaryEmbedding.from_float(model.value_embeds[k].weight)
    if cfg.lm_head:
        model.lm_head = TernaryLinear.from_float(model.lm_head.weight, quant_act=cfg.quant_act, act_qmax=cfg.act_qmax)
    if cfg.gates:
        model.smear_gate = TernaryLinear.from_float(model.smear_gate.weight, quant_act=cfg.quant_act, act_qmax=cfg.act_qmax)
        for block in model.transformer.h:
            if block.attn.ve_gate is not None:
                block.attn.ve_gate = TernaryLinear.from_float(block.attn.ve_gate.weight, quant_act=cfg.quant_act, act_qmax=cfg.act_qmax)
    model.ternary_config = asdict(cfg)
    return model


def ternary_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, _TernaryBase)]


def set_capture_grad(model, on: bool):
    for _, m in ternary_modules(model):
        m.capture_grad = on
        if not on:
            m.weight_grad = None


def decimal_param_report(model):
    """Every float parameter that is NOT a ternary matrix: name -> number of values."""
    return {n: p.numel() for n, p in model.named_parameters()}


def param_report(model):
    """Counts of stored weights by kind, and their storage in bytes."""
    tern = sum(m.weight_t.numel() for _, m in ternary_modules(model))
    fp = sum(p.numel() for p in model.parameters())
    tern_bytes_int8 = tern                    # as stored here (1 byte each)
    tern_bytes_packed = tern * 1.58 / 8       # information-theoretic packing (log2(3) bits)
    return dict(ternary_weights=tern, float_params=fp,
                ternary_MB_int8=tern_bytes_int8 / 1e6, ternary_MB_packed_1p58bit=tern_bytes_packed / 1e6,
                float_MB_fp32=fp * 4 / 1e6)


def setup_optimizer_ternary(model, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                            weight_decay=0.0, scalar_lr=0.5, ternary_scale_lr=0.002):
    """Same groups and hyperparameters as GPT.setup_optimizer, except:
    - ternary int8 weights are buffers, so no optimizer ever touches them (Phase 2 flip rules do)
    - each ternary matrix's single float scale goes to its own AdamW group
    - empty groups (e.g. wte when it is ternary) are dropped."""
    model_dim = model.config.n_embd
    dmodel_lr_scale = (model_dim / 768) ** -0.5
    scale_params = [m.scale for _, m in ternary_modules(model)]
    scale_ids = {id(p) for p in scale_params}

    block_params = [p for p in model.transformer.h.parameters() if id(p) not in scale_ids]
    matrix_params = [p for p in block_params if p.ndim == 2]
    assert len(matrix_params) == len(block_params), "unexpected non-matrix param in blocks"
    value_embeds_params = [p for p in model.value_embeds.parameters() if id(p) not in scale_ids]
    embedding_params = [p for p in model.transformer.wte.parameters() if id(p) not in scale_ids]
    lm_head_params = [p for p in model.lm_head.parameters() if id(p) not in scale_ids]
    smear_params = [p for p in model.smear_gate.parameters() if id(p) not in scale_ids] + [model.smear_lambda, model.backout_lambda]

    groups = [
        dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
        dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        dict(kind='adamw', params=[model.resid_lambdas], lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
        dict(kind='adamw', params=[model.x0_lambdas], lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind='adamw', params=scale_params, lr=ternary_scale_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    for shape in sorted({p.shape for p in matrix_params}):
        groups.append(dict(kind='muon', params=[p for p in matrix_params if p.shape == shape], lr=matrix_lr,
                           momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay))
    groups = [g for g in groups if len(g["params"]) > 0]

    n_in_groups = sum(len(g["params"]) for g in groups)
    assert n_in_groups == len(list(model.parameters())), f"{n_in_groups} vs {len(list(model.parameters()))}"
    opt = MuonAdamW(groups)
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    return opt
