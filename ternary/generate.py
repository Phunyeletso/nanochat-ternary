"""
Generate text from a model saved by ternary/train.py.
    python -m ternary.generate runs/ternary/<tag>.pt --prompt "It was a dark" --tokens 120
Works for all three kinds: full precision (A), BitNet with latent weights (B), direct ternary (C).
For C the weights used are exactly the stored int8 {-1,0,+1} values and one scale per matrix.
"""
import argparse

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from ternary.model import TernaryConfig, convert_to_ternary
from ternary.bitnet import convert_to_bitnet
from ternary.layers import qmax_for_tables


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    a = ck["args"]
    with torch.device("meta"):
        model = GPT(GPTConfig(**ck["gpt_config"]))
    model.to_empty(device=device)
    model.init_weights()
    if a.get("bitnet"):
        convert_to_bitnet(model)
    A_ = bool(a.get("ternary_all"))
    tcfg = TernaryConfig(linear=bool(a.get("ternary_linear")) or A_, embed=bool(a.get("ternary_embed")) or A_,
                         value_embeds=bool(a.get("ternary_value_embeds")) or A_, quant_act=bool(a.get("quant_act", 1)),
                         lm_head=bool(a.get("ternary_lm_head")) or A_, gates=bool(a.get("ternary_gates")) or A_,
                         act_qmax=qmax_for_tables(a.get("act_tables", 0)))
    if tcfg.linear or tcfg.embed or tcfg.value_embeds or tcfg.lm_head or tcfg.gates:
        convert_to_ternary(model, tcfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--prompt", action="append", default=None, help="can be given several times")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tok = get_tokenizer()
    model, a = load(args.ckpt)
    prompts = args.prompt or ["It was a", "The king said"]
    print(f"model: {a['tag']}")
    for pr in prompts:
        ids = tok(pr, prepend="<|bos|>")
        out = []
        with torch.no_grad():
            for t in model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k, seed=args.seed):
                if t == tok.get_bos_token_id():
                    break
                out.append(t)
        print(f"\n--- prompt: {pr!r}\n{pr}{tok.decode(out)}")


if __name__ == "__main__":
    main()
