#!/usr/bin/env python3
"""
PickScore aesthetic curation (the Krea "quality over quantity" lever, on pixels).

Scores every generated image against its own prompt with PickScore (a *human-preference*
model — deliberately NOT a LAION-aesthetic predictor, which Krea blames for the soft/
symmetric AI look), then keeps the top-`--keep` seeds per prompt above a score floor.

Operates on a local image dir written by batch_infer.py (WebDataset shards + manifest.jsonl).

    python select_aesthetic.py --image-dir out/images --keep 2 --out scored_manifest.jsonl
"""
import argparse, glob, io, json, os, tarfile
from collections import defaultdict
import torch
from PIL import Image

PICK_MODEL = "yuvalkirstain/PickScore_v1"
PICK_PROC  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def iter_images(image_dir):
    """Yield (key, prompt, PIL.Image) from all shards in image_dir."""
    prompts = {}
    for tar in sorted(glob.glob(os.path.join(image_dir, "*.tar"))):
        t = tarfile.open(tar)
        members = {m.name: m for m in t.getmembers()}
        for name, m in members.items():
            if name.endswith(".txt"):
                prompts[name[:-4]] = t.extractfile(m).read().decode("utf-8")
        for name, m in members.items():
            if name.endswith(".png"):
                key = name[:-4]
                img = Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
                yield key, prompts.get(key, ""), img
        t.close()


@torch.no_grad()
def score_all(image_dir, batch=32):
    from transformers import AutoModel, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(PICK_PROC)
    model = AutoModel.from_pretrained(PICK_MODEL).eval().to(dev)
    scores = {}
    buf_keys, buf_imgs, buf_txt = [], [], []

    def flush():
        if not buf_keys:
            return
        ii = proc(images=buf_imgs, return_tensors="pt").to(dev)
        ti = proc(text=buf_txt, padding=True, truncation=True, max_length=77,
                  return_tensors="pt").to(dev)
        ie = model.get_image_features(**ii); ie = ie / ie.norm(dim=-1, keepdim=True)
        te = model.get_text_features(**ti);  te = te / te.norm(dim=-1, keepdim=True)
        s = (model.logit_scale.exp() * (ie * te).sum(-1)).float().cpu()
        for k, v in zip(buf_keys, s):
            scores[k] = float(v)
        buf_keys.clear(); buf_imgs.clear(); buf_txt.clear()

    for key, prompt, img in iter_images(image_dir):
        buf_keys.append(key); buf_imgs.append(img); buf_txt.append(prompt or " ")
        if len(buf_keys) >= batch:
            flush()
    flush()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out", default="scored_manifest.jsonl")
    ap.add_argument("--keep", type=int, default=2, help="max seeds to keep per prompt")
    ap.add_argument("--floor-percentile", type=float, default=15.0,
                    help="drop images below this global score percentile (permissive for transfer)")
    args = ap.parse_args()

    scores = score_all(args.image_dir)
    print(f"[score] {len(scores)} images scored", flush=True)
    import numpy as np
    vals = np.array(list(scores.values()))
    floor = float(np.percentile(vals, args.floor_percentile))
    print(f"[score] mean={vals.mean():.2f} p15={np.percentile(vals,15):.2f} "
          f"p50={np.percentile(vals,50):.2f} p85={np.percentile(vals,85):.2f} floor={floor:.2f}")

    # group seeds by prompt id (key = <id>_s<k>)
    by_prompt = defaultdict(list)
    for key, sc in scores.items():
        pid = key.split("_s")[0]
        by_prompt[pid].append((key, sc))

    kept = 0
    with open(args.out, "w", encoding="utf-8") as o:
        for pid, lst in by_prompt.items():
            lst.sort(key=lambda x: -x[1])
            keep = [k for k, s in lst[:args.keep] if s >= floor]
            o.write(json.dumps({"id": pid, "keep": keep,
                                "scores": {k: round(s, 3) for k, s in lst}}) + "\n")
            kept += len(keep)
    print(f"[select] kept {kept} images across {len(by_prompt)} prompts "
          f"(<= {args.keep}/prompt, >= floor) -> {args.out}")


if __name__ == "__main__":
    main()
