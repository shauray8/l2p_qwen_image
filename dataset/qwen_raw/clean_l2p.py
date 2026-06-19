#!/usr/bin/env python3
"""
L2P "clean to the bones" — unified curation over the full corpus (new run + part0 +
l2p-dataset). ONE GPU pass computes PickScore + pHash + image embedding per image, then:

  1. per-prompt seed selection: keep top-`--keep` seeds whose PickScore >= global floor
     (permissive for the distillation transfer — cut only clear failures);
  2. pHash near-dup removal across survivors (cross-prompt look-alikes), vectorized;
  3. diversity audit: k-means over survivors -> inverse-freq sampling weights;
  4. repack survivors into WebDataset shards + manifest + sampling_weights.jsonl;
  5. push the cleaned set to --repo.

    python clean_l2p.py --src /workspace/clean/l2p-raw --src <part0_dir> --src <l2p_dataset_dir> \
        --out-dir /workspace/clean/final --repo shauray/l2p-dataset --keep 2 --push
"""
import argparse, glob, io, json, os, tarfile
import numpy as np
import torch
from PIL import Image

torch.backends.cuda.enable_cudnn_sdp(False)   # cuDNN SDPA broken on torch2.11/cu130 (CLIP-H would crash)

PICK_MODEL = "yuvalkirstain/PickScore_v1"
PICK_PROC  = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


def _split_src(s):
    """'prefix=dir' -> (prefix, dir); bare 'dir' -> ('', dir)."""
    if "=" in s and not os.path.exists(s):
        pre, d = s.split("=", 1); return pre + "_", d
    return "", s


def iter_shards(src_dirs):
    for s in src_dirs:
        prefix, d = _split_src(s)
        for tar in sorted(glob.glob(os.path.join(d, "*.tar"))):
            try:
                t = tarfile.open(tar)
                members = {m.name: m for m in t.getmembers()}
                txt = {k[:-4]: t.extractfile(m).read().decode("utf-8", "ignore")
                       for k, m in members.items() if k.endswith(".txt")}
                items = []
                for name, m in members.items():
                    if name.endswith(".png"):
                        items.append((prefix + name[:-4], txt.get(name[:-4], ""), t.extractfile(m).read()))
                t.close()
            except Exception as e:
                print(f"  [skip] corrupt tar {os.path.basename(tar)}: {type(e).__name__}", flush=True)
                continue
            for it in items:
                yield it


def phash_bits(img):
    size = 32
    im = img.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, np.float32)
    def dct(x):
        N = x.shape[0]; n = np.arange(N); k = n.reshape((N, 1))
        return np.cos(np.pi * (2 * n + 1) * k / (2 * N)) @ x
    d = dct(dct(a.T).T)[:8, :8]
    bits = (d > np.median(d[1:].flatten())).flatten().astype(np.uint8)
    return bits


@torch.no_grad()
def pass1(src_dirs, batch=32, limit=0):
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda"
    proc = CLIPProcessor.from_pretrained(PICK_PROC)
    model = CLIPModel.from_pretrained(PICK_MODEL).eval().to(dev)
    rows = []                      # {key, prompt, score, bits, emb}
    bk, bi, bt = [], [], []

    def flush():
        if not bk: return
        ii = proc(images=bi, return_tensors="pt").to(dev)
        ti = proc(text=bt, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
        # canonical CLIP/PickScore embeddings (version-stable: get_*_features changed return type in tf5.x)
        vis = model.vision_model(pixel_values=ii["pixel_values"]).pooler_output
        ie = model.visual_projection(vis); ie = ie / ie.norm(dim=-1, keepdim=True)
        txt = model.text_model(input_ids=ti["input_ids"],
                               attention_mask=ti.get("attention_mask")).pooler_output
        te = model.text_projection(txt); te = te / te.norm(dim=-1, keepdim=True)
        sc = (model.logit_scale.exp() * (ie * te).sum(-1)).float().cpu().numpy()
        emb = ie.float().cpu().numpy()
        for j, key in enumerate(bk):
            rows.append({"key": key, "score": float(sc[j]), "emb": emb[j]})
        bk.clear(); bi.clear(); bt.clear()

    meta = {}
    for key, prompt, png in iter_shards(src_dirs):
        img = Image.open(io.BytesIO(png)).convert("RGB")
        meta[key] = {"prompt": prompt, "bits": phash_bits(img)}
        bk.append(key); bi.append(img); bt.append(prompt or " ")
        if len(bk) >= batch: flush()
        if len(rows) and len(rows) % 2000 == 0:
            print(f"  scored {len(rows)}", flush=True)
        if limit and len(rows) + len(bk) >= limit: break
    flush()
    for r in rows:
        r.update(meta[r["key"]])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--out-dir", default="/workspace/clean/final")
    ap.add_argument("--repo", default="shauray/l2p-dataset")
    ap.add_argument("--keep", type=int, default=2, help="max seeds kept per prompt")
    ap.add_argument("--floor-percentile", type=float, default=12.0)
    ap.add_argument("--phash-dist", type=int, default=6)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--per-shard", type=int, default=200)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap images in pass1")
    args = ap.parse_args()

    rows = pass1(args.src, limit=args.limit)
    print(f"[pass1] {len(rows)} images scored", flush=True)
    scores = np.array([r["score"] for r in rows])
    floor = float(np.percentile(scores, args.floor_percentile))
    print(f"[floor] PickScore p{args.floor_percentile:.0f} = {floor:.2f} "
          f"(mean {scores.mean():.2f}, p50 {np.percentile(scores,50):.2f})")

    # 1) per-prompt best-k above floor
    from collections import defaultdict
    by_pid = defaultdict(list)
    for r in rows:
        by_pid[r["key"].split("_s")[0]].append(r)
    sel = []
    for pid, lst in by_pid.items():
        lst.sort(key=lambda r: -r["score"])
        sel += [r for r in lst[:args.keep] if r["score"] >= floor]
    print(f"[select] {len(sel)} images after per-prompt top-{args.keep} + floor "
          f"(from {len(by_pid)} prompts)")

    # 2) pHash dedup across survivors (vectorized Hamming)
    sel.sort(key=lambda r: -r["score"])      # keep the higher-scored of a near-dup pair
    kept, kept_bits = [], np.zeros((0, 64), np.uint8)
    for r in sel:
        b = r["bits"]
        if kept_bits.shape[0] and (kept_bits ^ b).sum(1).min() <= args.phash_dist:
            continue
        kept_bits = np.vstack([kept_bits, b]); kept.append(r)
    print(f"[phash] {len(kept)} images after near-dup removal (dropped {len(sel)-len(kept)})")

    # 3) diversity weights
    from sklearn.cluster import KMeans
    E = np.stack([r["emb"] for r in kept])
    k = min(args.k, max(2, len(kept) // 20))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(E)
    sizes = np.bincount(km.labels_, minlength=k).astype(float)
    shares = sizes / sizes.sum()
    cv = float(sizes.std() / sizes.mean())
    inv = 1.0 / sizes[km.labels_]; inv *= len(kept) / inv.sum()
    print(f"[diversity] k={k} CV={cv:.2f} largest={100*shares.max():.1f}% "
          f"({'GOOD' if cv<0.5 and shares.max()<0.04 else 'CHECK'})")

    keep_keys = {r["key"]: (int(km.labels_[i]), float(inv[i])) for i, r in enumerate(kept)}

    # 4) repack survivors (second pass over shards, read only kept) + weights
    os.makedirs(args.out_dir, exist_ok=True)
    man = open(os.path.join(args.out_dir, "manifest.jsonl"), "w", encoding="utf-8")
    wts = open(os.path.join(args.out_dir, "sampling_weights.jsonl"), "w", encoding="utf-8")
    sidx = cnt = 0; tar = None
    def newtar():
        nonlocal tar, sidx
        if tar: tar.close()
        tar = tarfile.open(os.path.join(args.out_dir, f"shard-{sidx:06d}.tar"), "w"); sidx += 1
    newtar()
    prompts_by_key = {r["key"]: r["prompt"] for r in kept}
    for key, prompt, png in iter_shards(args.src):
        if key not in keep_keys: continue
        if cnt and cnt % args.per_shard == 0: newtar()
        for nm, data in ((f"{key}.png", png), (f"{key}.txt", (prompt or "").encode())):
            ti = tarfile.TarInfo(nm); ti.size = len(data); tar.addfile(ti, io.BytesIO(data))
        cl, w = keep_keys[key]
        man.write(json.dumps({"id": key, "prompt": prompt, "files": [f"{key}.png"]}) + "\n")
        wts.write(json.dumps({"id": key, "cluster": cl, "weight": round(w, 4)}) + "\n")
        cnt += 1
    if tar: tar.close()
    man.close(); wts.close()
    print(f"[repack] {cnt} images -> {sidx} shards in {args.out_dir}")

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi(); api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=args.out_dir, repo_id=args.repo, repo_type="dataset",
                          commit_message=f"L2P cleaned set: {cnt} curated (prompt,image) pairs")
        print(f"[push] pushed {cnt} images -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
