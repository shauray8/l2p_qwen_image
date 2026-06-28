#!/usr/bin/env python3
"""Decoder-only reconstruction probe
Isolates the local-decoder head: take a clean training image, add a small amount
"""
import argparse, json, os, random, sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "l2p"))
sys.path.insert(0, os.path.dirname(__file__))
from qwen_image_l2p import QwenImageL2P  # noqa: E402
from dataset import OverfitDataset  # noqa: E402
import sample_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--weights", required=True,
                    help="full model safetensors (pixel-init or a step-N.safetensors export)")
    ap.add_argument("--transformer_config", default=None,
                    help="defaults to transformer_config.json next to --weights")
    ap.add_argument("--text_cache", default=None)
    ap.add_argument("--n", type=int, default=8, help="# images to probe")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2],
                    help="noise levels to sweep (sigma=0 is degenerate -> PSNR 100)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resize_base", type=int, default=0)
    ap.add_argument("--out", default=None, help="dir to dump gen|gt side-by-side pngs")
    ap.add_argument("--no_feature_use_norm_out", dest="feature_use_norm_out",
                    action="store_false", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_path = args.transformer_config or os.path.join(os.path.dirname(args.weights), "transformer_config.json")
    cfg = {k: v for k, v in json.load(open(cfg_path)).items() if not k.startswith("_")}

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = QwenImageL2P.from_config(cfg, feature_use_norm_out=args.feature_use_norm_out)
    torch.set_default_dtype(prev)
    model.load_state_dict(load_file(args.weights), strict=True, assign=False)
    model.to(device).eval()

    ds = OverfitDataset(args.data_dir, args.text_cache, repeat=1, limit=args.limit,
                        resize_base=args.resize_base)
    n = min(args.n, len(ds.rows))
    idxs = random.Random(args.seed).sample(range(len(ds.rows)), n)
    items = []
    for i in idxs:
        d = ds[i]
        img = ((d["image"].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()
        items.append({"image_np": img, "prompt_embeds": d["prompt_embeds"], "file_name": d["file_name"]})

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        from PIL import Image

    print(f"decoder recon | {n} items | weights={os.path.basename(args.weights)} | device={device.type}")
    for s in args.sigmas:
        psnr, lp, pairs = sample_eval.decoder_recon_metrics(model, items, device, sigma=s, seed=args.seed)
        msg = f"  sigma {s:.3f}: PSNR {psnr:6.2f}" + (f"  LPIPS {lp:.4f}" if lp is not None else "")
        print(msg, flush=True)
        if args.out:
            for gen, gt, fn in pairs:
                stem = os.path.splitext(os.path.basename(fn))[0]
                Image.fromarray(np.concatenate([gen, gt], axis=1)).save(
                    os.path.join(args.out, f"{stem}_s{s:.3f}.png"))


if __name__ == "__main__":
    main()
