#!/usr/bin/env python3
"""
Visual-manifold diversity audit (the real "is it just shrimp/bear/ocean?" check).
"""
import argparse, glob, io, json, os, tarfile
import numpy as np
import torch
from PIL import Image

CLIP = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

def iter_images(image_dir, only_keys=None):
    for tar in sorted(glob.glob(os.path.join(image_dir, "*.tar"))):
        t = tarfile.open(tar)
        for m in t.getmembers():
            if m.name.endswith(".png"):
                key = m.name[:-4]
                if only_keys is not None and key not in only_keys:
                    continue
                yield key, Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
        t.close()

@torch.no_grad()
def embed(image_dir, only_keys=None, batch=64):
    from transformers import AutoModel, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(CLIP)
    model = AutoModel.from_pretrained(CLIP).eval().to(dev)
    keys, embs, bk, bi = [], [], [], []

    def flush():
        if not bk:
            return
        ii = proc(images=bi, return_tensors="pt").to(dev)
        e = model.get_image_features(**ii)
        e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        embs.append(e); keys.extend(bk); bk.clear(); bi.clear()

    for key, img in iter_images(image_dir, only_keys):
        bk.append(key); bi.append(img)
        if len(bk) >= batch:
            flush()
    flush()
    return keys, (np.concatenate(embs, 0) if embs else np.zeros((0, 1024)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--keep-manifest", default=None,
                    help="scored_manifest.jsonl from select_aesthetic; audit only kept images")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--report", default="diversity_report.json")
    ap.add_argument("--weights", default="sampling_weights.jsonl")
    args = ap.parse_args()

    only = None
    if args.keep_manifest:
        only = set()
        for l in open(args.keep_manifest):
            only.update(json.loads(l).get("keep", []))

    keys, E = embed(args.image_dir, only)
    n = len(keys)
    print(f"[embed] {n} images", flush=True)
    from sklearn.cluster import KMeans
    k = min(args.k, max(2, n // 20))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(E)
    labels = km.labels_
    sizes = np.bincount(labels, minlength=k).astype(float)
    shares = sizes / sizes.sum()
    order = np.argsort(-sizes)
    report = {
        "n_images": int(n), "k": int(k),
        "cv": float(sizes.std() / sizes.mean()),
        "largest_mode_share": float(shares.max()),
        "top10_share": float(np.sort(shares)[::-1][:10].sum()),
        "effective_modes": float(1.0 / (shares ** 2).sum()),     # inverse Simpson
        "health": None,
    }
    report["health"] = ("GOOD" if report["cv"] < 0.5 and report["largest_mode_share"] < 0.04
                        else "CHECK")
    json.dump(report, open(args.report, "w"), indent=2)
    print(f"[audit] CV={report['cv']:.2f} largest={100*report['largest_mode_share']:.1f}% "
          f"top10={100*report['top10_share']:.1f}% eff_modes={report['effective_modes']:.0f} "
          f"-> {report['health']}")

    # inverse-frequency weights, normalized to mean 1.0
    inv = (1.0 / sizes[labels])
    inv *= n / inv.sum()
    with open(args.weights, "w", encoding="utf-8") as o:
        for key, lab, w in zip(keys, labels, inv):
            o.write(json.dumps({"id": key, "cluster": int(lab), "weight": round(float(w), 4)}) + "\n")
    print(f"[weights] wrote {n} sampling weights -> {args.weights}")

if __name__ == "__main__":
    main()
