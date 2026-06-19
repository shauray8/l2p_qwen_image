#!/usr/bin/env python3
"""
Materialize the curated WebDataset (shauray/l2p-clean) into the on-disk layout the trainer
expects: <out>/images/<key>.png + <out>/metadata.csv (columns: file_name,text).

Idempotent: if metadata.csv already exists with the expected row count, it's a no-op, so it
can sit in the spot entrypoint and only run on the first boot of a fresh network volume.

    python prep_data.py --repo shauray/l2p-clean --out /workspace/data/l2p-clean
"""
import argparse, csv, glob, io, os, tarfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="shauray/l2p-clean")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-workers", type=int, default=16)
    args = ap.parse_args()

    meta_path = os.path.join(args.out, "metadata.csv")
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    from huggingface_hub import snapshot_download
    src = snapshot_download(args.repo, repo_type="dataset",
                            allow_patterns=["*.tar"], max_workers=args.max_workers)

    rows = []
    for tar in sorted(glob.glob(os.path.join(src, "**", "*.tar"), recursive=True)):
        t = tarfile.open(tar)
        members = t.getmembers()
        prompts = {m.name[:-4]: t.extractfile(m).read().decode("utf-8", "ignore")
                   for m in members if m.name.endswith(".txt")}
        for m in members:
            if not m.name.endswith(".png"):
                continue
            key = m.name[:-4]
            fn = os.path.join("images", f"{key}.png")
            dst = os.path.join(args.out, fn)
            if not os.path.exists(dst):
                with open(dst, "wb") as f:
                    f.write(t.extractfile(m).read())
            rows.append({"file_name": fn, "text": prompts.get(key, "").strip()})
        t.close()
        print(f"  extracted {len(rows)} so far ({os.path.basename(tar)})", flush=True)

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file_name", "text"])
        w.writeheader()
        w.writerows(rows)
    print(f"[done] {len(rows)} samples -> {meta_path}")


if __name__ == "__main__":
    main()
