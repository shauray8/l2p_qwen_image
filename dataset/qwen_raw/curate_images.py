#!/usr/bin/env python3
"""
Image-level dedup + failure-floor curation for the L2P "clean to the bones" pass
(MD §8.1/§8.4). Runs over one or more WebDataset image dirs (new run + existing
part0/l2p-dataset shards), and produces a keep-list.

Two cuts, both PERMISSIVE for the distillation transfer (cut only clear duplicates /
failures, keep the teacher's manifold):
  1. pHash near-duplicate dedup — drop images whose perceptual hash is within
     --phash-dist of an already-kept one (catches identical-looking gens from
     different prompts/seeds that prompt-dedup can't see).
  2. (optional) PickScore floor — pass --scored scored_manifest.jsonl from
     select_aesthetic.py to also drop images below the global failure floor.

    python curate_images.py --image-dir /workspace/out/images_new \
        [--image-dir /path/to/part0_shards ...] --phash-dist 6 --out keep_images.jsonl
"""
import argparse, glob, io, json, os, tarfile
import numpy as np
from PIL import Image


def phash(img, hash_size=8, highfreq=4):
    """DCT-based perceptual hash -> 64-bit int."""
    size = hash_size * highfreq
    im = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    dct = _dct2(a)[:hash_size, :hash_size]
    med = np.median(dct[1:].flatten())
    bits = (dct > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _dct2(a):
    return _dct(_dct(a.T).T)


def _dct(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return M @ x


def iter_images(image_dirs):
    for d in image_dirs:
        for tar in sorted(glob.glob(os.path.join(d, "*.tar"))):
            t = tarfile.open(tar)
            for m in t.getmembers():
                if m.name.endswith(".png"):
                    yield m.name[:-4], Image.open(io.BytesIO(t.extractfile(m).read())).convert("RGB")
            t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", action="append", required=True)
    ap.add_argument("--phash-dist", type=int, default=6, help="Hamming dist <= this = duplicate")
    ap.add_argument("--scored", default=None, help="scored_manifest.jsonl (PickScore) for failure floor")
    ap.add_argument("--keys", default=None, help="jsonl of {key} to restrict to (e.g. PickScore-selected seeds)")
    ap.add_argument("--floor-percentile", type=float, default=10.0)
    ap.add_argument("--out", default="keep_images.jsonl")
    args = ap.parse_args()

    floor_drop = set()
    if args.scored:
        scores = {}
        for l in open(args.scored):
            r = json.loads(l)
            for k, s in r.get("scores", {}).items():
                scores[k] = s
        if scores:
            thr = np.percentile(list(scores.values()), args.floor_percentile)
            floor_drop = {k for k, s in scores.items() if s < thr}
            print(f"[floor] PickScore p{args.floor_percentile:.0f}={thr:.2f} -> drop {len(floor_drop)}")

    only = None
    if args.keys:
        only = {json.loads(l).get("key") for l in open(args.keys) if l.strip()}
        print(f"[keys] restricting pHash to {len(only)} pre-selected images")

    kept, dropped_dup, dropped_floor = [], 0, 0
    kept_bits = np.zeros((0, 64), dtype=np.uint8)   # vectorized Hamming over kept hashes
    n = 0
    for key, img in iter_images(args.image_dir):
        if only is not None and key not in only:
            continue
        n += 1
        if key in floor_drop:
            dropped_floor += 1
            continue
        bits = np.unpackbits(np.frombuffer(phash(img).to_bytes(8, "big"), dtype=np.uint8))
        if kept_bits.shape[0]:
            hd = (kept_bits ^ bits).sum(1)        # Hamming distance to every kept hash
            if hd.min() <= args.phash_dist:
                dropped_dup += 1
                continue
        kept_bits = np.vstack([kept_bits, bits])
        kept.append(key)
        if n % 2000 == 0:
            print(f"  scanned {n} | kept {len(kept)} | dup {dropped_dup} | floor {dropped_floor}", flush=True)

    with open(args.out, "w", encoding="utf-8") as o:
        for k in kept:
            o.write(json.dumps({"key": k}) + "\n")
    print(f"[done] scanned {n} | KEPT {len(kept)} | dropped dup {dropped_dup} | floor {dropped_floor} -> {args.out}")


if __name__ == "__main__":
    main()
