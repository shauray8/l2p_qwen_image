#!/usr/bin/env python3
"""
Convert pretrained Qwen-Image-2512 (latent) transformer weights into an L2P
pixel-space initialization.

Output is a single safetensors holding the full QwenImageL2P state_dict
"""
import argparse, json, os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import QwenImageTransformer2DModel

from qwen_image_l2p import QwenImageL2P, IN_CH, PIXEL_PATCH


def load_source_transformer_sd(model_id_or_path: str):
    if os.path.isdir(model_id_or_path):
        tdir = os.path.join(model_id_or_path, "transformer")
        cfg = json.load(open(os.path.join(tdir, "config.json")))
        files = [f for f in os.listdir(tdir) if f.endswith(".safetensors")]
    else:
        cfg_path = hf_hub_download(model_id_or_path, "transformer/config.json")
        cfg = json.load(open(cfg_path))
        d = snapshot_download(model_id_or_path, allow_patterns=["transformer/*.safetensors"])
        tdir = os.path.join(d, "transformer")
        files = [f for f in os.listdir(tdir) if f.endswith(".safetensors")]
    sd = {}
    for f in sorted(files):
        sd.update(load_file(os.path.join(tdir, f)))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return sd, cfg

def convert(model_id_or_path, output_path, feature_use_norm_out=False, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    print(f"[load] source transformer from {model_id_or_path} ...")
    source, cfg = load_source_transformer_sd(model_id_or_path)
    print(f"[load] {len(source)} source tensors | config: { {k: cfg[k] for k in ('in_channels','num_layers','num_attention_heads','attention_head_dim','joint_attention_dim')} }")

    with torch.device("meta"):
        meta_model = QwenImageL2P.from_config(cfg, feature_use_norm_out=feature_use_norm_out)
    target_shapes = {k: tuple(v.shape) for k, v in meta_model.state_dict().items()}
    print(f"[target] {len(target_shapes)} pixel-model tensors")

    new_init = {}
    img_in = nn.Linear(IN_CH * PIXEL_PATCH * PIXEL_PATCH, cfg["num_attention_heads"] * cfg["attention_head_dim"])
    for k, v in img_in.state_dict().items():
        new_init[f"dit.img_in.{k}"] = v
    from local_decoder import MicroDiffusionModel
    dec = MicroDiffusionModel(IN_CH, cfg["num_attention_heads"] * cfg["attention_head_dim"])
    for k, v in dec.state_dict().items():
        new_init[f"local_decoder.{k}"] = v

    final = {}
    copied = reinit = missing = 0
    for tk, tshape in target_shapes.items():
        if tk in new_init:
            final[tk] = new_init[tk].to(dtype)
            reinit += 1
            continue
        assert tk.startswith("dit."), tk
        sk = tk[len("dit."):]
        if sk in source and tuple(source[sk].shape) == tshape:
            final[tk] = source[sk].to(dtype)    
            copied += 1
        else:
            print(f"no source match for {tk} (shape {tshape}); zero-init")
            final[tk] = torch.zeros(tshape, dtype=dtype)
            missing += 1

    print(f"[migrate] copied={copied} reinit(new)={reinit} fallback={missing} -> {len(final)} tensors")
    assert set(final) == set(target_shapes), "key mismatch vs pixel model state_dict"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    meta = {"format": "pt", "l2p_in_channels": str(IN_CH * PIXEL_PATCH * PIXEL_PATCH),
            "feature_use_norm_out": str(feature_use_norm_out)}
    save_file(final, output_path, metadata=meta)
    json.dump(cfg, open(os.path.join(os.path.dirname(os.path.abspath(output_path)), "transformer_config.json"), "w"), indent=1)
    print(f"[save] {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2512", help="HF id or local dir of Qwen-Image")
    ap.add_argument("--output", default="./pretrain_weight/Qwen-Image-Pixel-Init/model.safetensors")
    ap.add_argument("--feature_use_norm_out", action="store_true",
                    help="apply norm_out to the image-stream feature map (default: skip, like Z-Image L2P)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    convert(args.model, args.output, feature_use_norm_out=args.feature_use_norm_out, seed=args.seed)
