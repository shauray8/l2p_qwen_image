#!/usr/bin/env python3
"""
Build an anatomy-grounded prompt pool from shauray/aesthetic-cleaned-captioned.
"""
import argparse, json, os, re
from collections import defaultdict
from huggingface_hub import hf_hub_download

HUMAN = re.compile(r"\b(woman|man|men|women|person|people|girl|boy|portrait|figure|"
                   r"body|nude|model|dancer|athlete|pose|posing|couple|child|hands?|"
                   r"standing|sitting|seated|kneeling|reclining)\b", re.I)
# pose / anatomy words make a caption especially valuable for the anatomy goal
POSE = re.compile(r"\b(pose|posing|hand|arm|leg|shoulder|hip|knee|elbow|back|spine|"
                  r"turned|facing|leaning|stretch|bent|crossed|contrapposto|gesture|"
                  r"reclining|kneeling|seated|standing|barefoot|stride|profile)\b", re.I)

def fit_length(text, lo=250, hi=450):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= hi:
        return text if len(text) >= lo else None
    out = ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(sent) + 1 > hi:
            break
        out = (out + " " + sent).strip()
    return out if lo <= len(out) <= hi else (text[:hi].rsplit(" ", 1)[0] if len(text) > lo else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="shauray/aesthetic-cleaned-captioned")
    ap.add_argument("--out", default="prompts_anatomy.jsonl")
    ap.add_argument("--max", type=int, default=1200, help="max anatomy prompts to keep")
    ap.add_argument("--per-photographer", type=int, default=8)
    args = ap.parse_args()

    p = hf_hub_download(args.repo, "final_manifest.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p)]

    seen_group, per_phot = set(), defaultdict(int)
    cands = []
    for r in rows:
        cap = (r.get("caption") or "").strip()
        if not cap:
            continue
        if not HUMAN.search(cap):
            continue
        grp = r.get("dup_group")
        if grp is not None and grp in seen_group:
            continue
        phot = r.get("photographer", "")
        if per_phot[phot] >= args.per_photographer:
            continue
        prompt = fit_length(cap)
        if not prompt:
            continue
        if grp is not None:
            seen_group.add(grp)
        per_phot[phot] += 1
        # pose-rich captions sort first (strongest anatomy value)
        cands.append((bool(POSE.search(cap)), len(POSE.findall(cap)), {
            "super_class": "People", "sub_class": "anatomy_real",
            "subject": phot, "source": "pexels-caption",
            "prompt": prompt}))

    cands.sort(key=lambda x: (x[0], x[1]), reverse=True)
    kept = [c[2] for c in cands[:args.max]]
    with open(args.out, "w", encoding="utf-8") as o:
        for rec in kept:
            o.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)} anatomy prompts (from {len(cands)} candidates) -> {args.out}")

if __name__ == "__main__":
    main()
