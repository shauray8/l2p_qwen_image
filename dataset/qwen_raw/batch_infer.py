#!/usr/bin/env python3
"""
Stage-D image generation with Qwen-Image-2512 (the L2P teacher).

Reads a prompts jsonl ({id?, prompt, super_class?, sub_class?, seed_start?}), generates
`--seeds` images per prompt at content-aware aspect buckets (~1.6 MP, Qwen-native), and
writes WebDataset shards (<id>_s<k>.png + <id>_s<k>.txt, 200 imgs/shard) + manifest.jsonl
— matching the existing l2p-dataset / part0 layout so shards append cleanly.

Attention backend is pluggable for the speed benchmark:
  --attn sdpa  : torch SDPA forced to the FLASH backend (cuDNN SDPA is broken on this box)
  --attn fa3   : FlashAttention-3 via dataset/fa3_loader (Hopper), Qwen double-stream processor

  python batch_infer.py --prompts prompts_final.jsonl --out-dir out/images \
      --seeds 3 --steps 28 --true-cfg 4.0 --attn fa3 --magic
"""
import argparse, io, json, os, sys, tarfile, time
import torch

MAGIC = " Ultra HD, 4K, cinematic composition."

# 7 Qwen-native aspect buckets at ~1.6 MP (w, h), multiples of 16.
BUCKETS = {
    "1:1":  (1328, 1328),
    "4:3":  (1472, 1104),
    "3:4":  (1104, 1472),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "3:2":  (1584, 1056),
    "2:3":  (1056, 1584),
}
WIDE = {"Landscape", "Cityscape", "Sports", "Others"}
TALL = {"Portrait", "anatomy_real"}
SQUARE_DEFAULT = "1:1"


def pick_bucket(sub_class, prompt):
    s = (sub_class or "")
    p = (prompt or "").lower()
    if s in TALL or any(w in p for w in ("portrait", "standing", "full-body", "full body")):
        return "3:4"
    if s in WIDE or any(w in p for w in ("landscape", "vista", "panorama", "skyline", "wide establishing")):
        return "16:9"
    if s in ("Food", "Objects", "Plants", "Arts"):
        return SQUARE_DEFAULT
    return SQUARE_DEFAULT


def setup_attention(pipe, mode):
    # cuDNN SDPA plan is broken under torch 2.11/cu130 on this box — disable it GLOBALLY so
    # the text encoder (Qwen2.5-VL, via transformers SDPA) doesn't crash, regardless of the
    # diffusion-transformer backend we pick below.
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    if mode == "sdpa":
        return "sdpa(flash)"
    if mode == "fa3":
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from fa3_loader import load_fa3
        fa3 = load_fa3()
        _install_fa3_dispatch(fa3)
        return "fa3"
    raise ValueError(mode)


def _install_fa3_dispatch(fa3):
    """Route Qwen's joint attention through FA3.

    The Qwen double-stream processor calls dispatch_attention_fn(q,k,v, attn_mask, ...)
    with q,k,v already in [B, S, H, D] — FA3's native layout. For T2I inference there is
    no attention mask, so fa3.flash_attn_func(q,k,v) is a drop-in. We patch the symbol in
    the transformer module's namespace and fall back to the original when a mask is present.
    """
    import diffusers.models.transformers.transformer_qwenimage as tq
    orig = tq.dispatch_attention_fn

    def fa3_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                     is_causal=False, backend=None, parallel_config=None, **kw):
        if attn_mask is None and not is_causal and query.dtype in (torch.bfloat16, torch.float16):
            o = fa3.flash_attn_func(query.contiguous(), key.contiguous(), value.contiguous())
            return o[0] if isinstance(o, tuple) else o
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, backend=backend, parallel_config=parallel_config, **kw)

    tq.dispatch_attention_fn = fa3_dispatch


def load_prompts(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for i, r in enumerate(rows):
        r.setdefault("id", f"{i:08d}")
    return rows


class ShardWriter:
    def __init__(self, out_dir, per_shard=200, start_index=0):
        self.dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.per = per_shard; self.idx = start_index; self.count = 0; self.tar = None
        self.manifest = open(os.path.join(out_dir, "manifest.jsonl"), "a", encoding="utf-8")
        self._open()

    def _open(self):
        self.tar = tarfile.open(os.path.join(self.dir, f"shard-{self.idx:06d}.tar"), "w")

    def _add(self, name, data: bytes):
        ti = tarfile.TarInfo(name); ti.size = len(data)
        self.tar.addfile(ti, io.BytesIO(data))

    def write(self, key, img_png: bytes, prompt: str):
        if self.count and self.count % self.per == 0:
            self.tar.close(); self.idx += 1; self._open()
        self._add(f"{key}.png", img_png)
        self._add(f"{key}.txt", prompt.encode("utf-8"))
        self.count += 1

    def record(self, rec):
        self.manifest.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.manifest.flush()

    def close(self):
        if self.tar: self.tar.close()
        self.manifest.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2512")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", default="out/images")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0,
                    help="seed index offset (use to add seeds to prompts that already have some)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--attn", choices=["sdpa", "fa3"], default="fa3")
    ap.add_argument("--magic", action="store_true")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--per-shard", type=int, default=200)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap prompts (smoke test)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the transformer")
    ap.add_argument("--cache", action="store_true", help="enable Cache-DiT (DBCache) acceleration")
    args = ap.parse_args()

    from diffusers import QwenImagePipeline
    print(f"[init] loading {args.model} ...", flush=True)
    t0 = time.time()
    pipe = QwenImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    backend = setup_attention(pipe, args.attn)
    if args.cache:
        import cache_dit
        cache_dit.enable_cache(pipe)
        backend += "+cache-dit"
    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer, fullgraph=False)
    print(f"[init] ready in {time.time()-t0:.0f}s | attn={backend}", flush=True)

    rows = load_prompts(args.prompts)
    if args.limit:
        rows = rows[:args.limit]
    writer = ShardWriter(args.out_dir, args.per_shard, args.shard_start)

    neg = " "
    done = 0; t1 = time.time()
    for r in rows:
        base = (r["prompt"] + MAGIC) if args.magic else r["prompt"]
        bucket = pick_bucket(r.get("sub_class"), r["prompt"])
        w, h = BUCKETS[bucket]
        files = []
        for k in range(args.seeds):
            sidx = args.seed_base + k
            g = torch.Generator(device="cuda").manual_seed(hash((r["id"], sidx)) & 0x7fffffff)
            img = pipe(prompt=base, negative_prompt=neg, width=w, height=h,
                       num_inference_steps=args.steps, true_cfg_scale=args.true_cfg,
                       generator=g).images[0]
            buf = io.BytesIO(); img.save(buf, format="PNG")
            key = f"{r['id']}_s{sidx}"
            writer.write(key, buf.getvalue(), r["prompt"])
            files.append(f"{key}.png")
            done += 1
        writer.record({"id": r["id"], "prompt": r["prompt"], "resolution": f"{w}x{h}",
                       "super_class": r.get("super_class"), "sub_class": r.get("sub_class"),
                       "files": files})
        if done % 20 < args.seeds:
            dt = time.time() - t1
            print(f"[gen] {done} imgs | {done/max(dt,1):.2f} img/s | {dt:.0f}s", flush=True)
    writer.close()
    print(f"\n==== DONE ==== {done} images in {time.time()-t1:.0f}s -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
