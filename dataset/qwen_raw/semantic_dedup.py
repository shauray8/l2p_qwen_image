#!/usr/bin/env python3
"""
SemDeDup-style semantic dedup + coverage check (mirrors the MD's semantic_dedup.py).

- Embed every prompt with bge-small-en-v1.5 (mean-pooled, L2-normalized).
- Greedily drop near-duplicates at cosine > --threshold (default 0.90).
- Optionally drop prompts too close to an --against set (existing corpus), so newly
  generated prompts don't repeat what's already on the Hub.
- k-means coverage report: cluster-size CV + largest cluster share.

    python semantic_dedup.py --in prompts_clean.jsonl --out prompts_final.jsonl \
        --threshold 0.90 --against audit/existing_prompts.jsonl
"""
import argparse, json, sys
import numpy as np


def embed(prompts, batch=256, model_name="BAAI/bge-small-en-v1.5"):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # torch 2.11/cu130 + cudnn 9.19 has a broken cuDNN SDPA plan on this box; bge is tiny
    # so eager attention is both safe and fast.
    torch.backends.cuda.enable_cudnn_sdp(False)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(dev).eval().half()
    out = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=192,
                      return_tensors="pt").to(dev)
            h = mdl(**enc).last_hidden_state            # [B, T, H]
            mask = enc.attention_mask.unsqueeze(-1).half()
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1)   # mean pool
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.float().cpu().numpy())
            print(f"\r  embedding {min(i+batch,len(prompts))}/{len(prompts)}",
                  end="", file=sys.stderr)
    print("", file=sys.stderr)
    return np.concatenate(out, 0)


def greedy_dedup(E, thr, ref=None, chunk=2048):
    """Keep mask over rows of E; drop any row whose max cosine to an already-kept row
    (or to any ref row) exceeds thr."""
    n = E.shape[0]
    keep = np.ones(n, bool)
    # drop vs reference (existing) set first
    if ref is not None and len(ref):
        for i in range(0, n, chunk):
            sims = E[i:i+chunk] @ ref.T
            hit = sims.max(1) > thr
            keep[i:i+chunk][hit] = False
    # greedy within-set: walk in order, compare to running kept set
    kept_idx = []
    kept_vecs = []
    for i in range(n):
        if not keep[i]:
            continue
        if kept_vecs:
            K = np.asarray(kept_vecs)
            if (E[i] @ K.T).max() > thr:
                keep[i] = False
                continue
        kept_idx.append(i); kept_vecs.append(E[i])
    return keep


def coverage(E, k=200):
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None
    k = min(k, max(2, E.shape[0] // 20))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(E)
    sizes = np.bincount(km.labels_, minlength=k).astype(float)
    return {"k": int(k), "cv": float(sizes.std() / sizes.mean()),
            "largest_share": float(sizes.max() / sizes.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--against", default=None, help="existing-corpus jsonl to dedup against")
    ap.add_argument("--k", type=int, default=200, help="k for coverage report")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8") if l.strip()]
    prompts = [r[args.field] for r in rows]
    print(f"[in] {len(prompts)} prompts", file=sys.stderr)
    E = embed(prompts)

    ref = None
    if args.against:
        ref_rows = [json.loads(l) for l in open(args.against, encoding="utf-8") if l.strip()]
        ref_prompts = [r.get(args.field, "") for r in ref_rows if r.get(args.field)]
        print(f"[against] {len(ref_prompts)} existing prompts", file=sys.stderr)
        ref = embed(ref_prompts)

    keep = greedy_dedup(E, args.threshold, ref=ref)
    with open(args.out, "w", encoding="utf-8") as o:
        for r, k in zip(rows, keep):
            if k:
                o.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept = int(keep.sum())
    cov = coverage(E[keep], args.k)
    print(f"[out] kept {kept}/{len(prompts)} "
          f"(dropped {len(prompts)-kept} near-dups @cos>{args.threshold}) -> {args.out}",
          file=sys.stderr)
    if cov:
        print(f"[coverage] k={cov['k']} CV={cov['cv']:.2f} "
              f"largest_cluster={100*cov['largest_share']:.1f}%  "
              f"({'GOOD' if cov['cv']<0.5 and cov['largest_share']<0.04 else 'CHECK'})",
              file=sys.stderr)


if __name__ == "__main__":
    main()
