#!/usr/bin/env python3
"""
Stage-D production image generation with Qwen-Image-2512 via SGLang's native backend.

TOP-NOTCH quality config (no approximation): SGLang native DiT + FA3, bf16, 28 steps,
true_cfg_scale 4.0, NO cache-dit. ~13.6 s/img on one H100 — the SGLang speedup over
diffusers is from FA3/optimized kernels (lossless), not step-skipping.

Reads a worklist jsonl: {id, prompt, super_class, sub_class, seeds:[k,...]} where `seeds`
are the seed indices to (re)generate for that prompt. Writes WebDataset shards
(<id>_s<k>.png + <id>_s<k>.txt, 200 imgs/shard) + manifest.jsonl, matching the existing
l2p-dataset/part0 layout. RESUMABLE: skips <id>_s<k> already present in the manifest.

    # quality (default): no cache
    python gen_sglang.py --worklist worklist.jsonl --out-dir out/images
    # opt-in speed (lossy, NOT default): aggressive cache
    SGLANG_CACHE_DIT_ENABLED=1 SGLANG_CACHE_DIT_RDT=0.24 python gen_sglang.py ... --allow-cache
"""
import argparse, io, json, os, sys, tarfile, time
import numpy as np

MAGIC = " Ultra HD, 4K, cinematic composition."
BUCKETS = {"1:1": (1328, 1328), "4:3": (1472, 1104), "3:4": (1104, 1472),
           "16:9": (1664, 928), "9:16": (928, 1664), "3:2": (1584, 1056), "2:3": (1056, 1584)}
WIDE = {"Landscape", "Cityscape", "Sports", "Others"}
TALL = {"Portrait", "anatomy_real"}


def pick_bucket(sub_class, prompt):
    s, p = (sub_class or ""), (prompt or "").lower()
    if s in TALL or any(w in p for w in ("portrait", "standing", "full-body", "full body", "figure")):
        return "3:4"
    if s in WIDE or any(w in p for w in ("landscape", "vista", "panorama", "skyline", "establishing")):
        return "16:9"
    return "1:1"


def to_pil(samples):
    from PIL import Image
    x = samples
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    a = np.asarray(x)
    if a.ndim == 4:
        a = a[0]
    if a.dtype != np.uint8:
        a = (a.clip(0, 1) * 255).round().astype(np.uint8) if a.max() <= 1.0 else a.astype(np.uint8)
    if a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):   # CHW -> HWC
        a = a.transpose(1, 2, 0)
    return Image.fromarray(a[..., :3]).convert("RGB")


class ShardWriter:
    def __init__(self, out_dir, per_shard=200):
        self.dir = out_dir; os.makedirs(out_dir, exist_ok=True)
        self.per = per_shard
        self.manifest_path = os.path.join(out_dir, "manifest.jsonl")
        self.done = set()
        if os.path.exists(self.manifest_path):
            for l in open(self.manifest_path):
                try:
                    self.done.update(json.loads(l).get("files", []))
                except Exception:
                    pass
        # resume shard index from existing shards
        import glob
        shards = sorted(glob.glob(os.path.join(out_dir, "shard-*.tar")))
        self.idx = (int(os.path.basename(shards[-1])[6:12]) + 1) if shards else 0
        self.count = 0; self.tar = None
        self.manifest = open(self.manifest_path, "a", encoding="utf-8")

    def _open(self):
        self.tar = tarfile.open(os.path.join(self.dir, f"shard-{self.idx:06d}.tar"), "w")

    def _add(self, name, data):
        ti = tarfile.TarInfo(name); ti.size = len(data); self.tar.addfile(ti, io.BytesIO(data))

    def write(self, key, png, prompt):
        if self.tar is None or (self.count and self.count % self.per == 0):
            if self.tar: self.tar.close()
            if self.count: self.idx += 1
            self._open()
        self._add(f"{key}.png", png); self._add(f"{key}.txt", prompt.encode("utf-8"))
        self.count += 1

    def record(self, rec): self.manifest.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.manifest.flush()
    def close(self):
        if self.tar: self.tar.close()
        self.manifest.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2512")
    ap.add_argument("--worklist", required=True)
    ap.add_argument("--out-dir", default="out/images")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--true-cfg", type=float, default=4.0)
    ap.add_argument("--magic", action="store_true", default=True)
    ap.add_argument("--per-shard", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-cache", action="store_true",
                    help="permit cache-dit if SGLANG_CACHE_DIT_ENABLED is set (default: force OFF for quality)")
    args = ap.parse_args()

    if not args.allow_cache:
        os.environ["SGLANG_CACHE_DIT_ENABLED"] = "0"   # guarantee no lossy step-skipping

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    sa = ServerArgs(model_path=args.model, backend="sglang", performance_mode="speed",
                    trust_remote_code=True)
    set_global_server_args(sa)
    print(f"[init] loading {args.model} (cache={'on' if args.allow_cache else 'OFF'}) ...", flush=True)
    t0 = time.time()
    engine = DiffGenerator.from_server_args(sa, local_mode=True)
    print(f"[init] ready in {time.time()-t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in open(args.worklist) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    writer = ShardWriter(args.out_dir, args.per_shard)

    done = 0; gen = 0; t1 = time.time()
    for r in rows:
        rid = r["id"]; prompt = r["prompt"]
        base = prompt + MAGIC if args.magic else prompt
        w, h = BUCKETS[pick_bucket(r.get("sub_class"), prompt)]
        files = []
        for k in r.get("seeds", [0, 1, 2]):
            key = f"{rid}_s{k}"
            if f"{key}.png" in writer.done:
                files.append(f"{key}.png"); continue
            seed = (hash((rid, k)) & 0x7fffffff)
            try:
                res = engine.generate(sampling_params_kwargs=dict(
                    prompt=base, negative_prompt=" ", height=h, width=w, seed=seed,
                    num_inference_steps=args.steps, true_cfg_scale=args.true_cfg,
                    save_output=False))
            except Exception as e:
                print(f"[err] {key}: {type(e).__name__}: {str(e)[:120]}", flush=True); continue
            res = res[0] if isinstance(res, list) else res
            img = to_pil(res.samples)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            writer.write(key, buf.getvalue(), prompt)
            files.append(f"{key}.png"); gen += 1; done += 1
            if gen % 25 == 0:
                dt = time.time() - t1
                print(f"[gen] {gen} new imgs | {dt/max(gen,1):.1f}s/img | {dt:.0f}s elapsed", flush=True)
        if files:
            writer.record({"id": rid, "prompt": prompt, "resolution": f"{w}x{h}",
                           "super_class": r.get("super_class"), "sub_class": r.get("sub_class"),
                           "files": files})
    writer.close()
    print(f"\n==== DONE ==== {gen} new images -> {args.out_dir} in {time.time()-t1:.0f}s", flush=True)


if __name__ == "__main__":
    main()
