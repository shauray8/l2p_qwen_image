#!/usr/bin/env python3
"""
Stage-C+ normalization: strip stray markdown / quotes / enumerators / fences left by
"""
import argparse, json, re, sys
from collections import Counter

REFUSAL = re.compile(r"\b(i('| a)m sorry|i cannot|i can't|cannot assist|as an ai|"
                     r"i am unable|unable to (help|assist)|against my guidelines)\b", re.I)
_ENUM   = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")          # "1. ", "- ", "* "
_FENCE  = re.compile(r"^\s*```.*?$", re.M)
_LABEL  = re.compile(r"^\s*(?:prompt|caption|image)\s*\d*\s*[:\-]\s*", re.I)

def clean_text(p: str) -> str:
    p = _FENCE.sub("", p)
    p = p.strip()
    # strip a single layer of wrapping quotes / backticks
    for a, b in (('"', '"'), ("'", "'"), ("`", "`"), ("“", "”")):
        if len(p) > 1 and p[0] == a and p[-1] == b:
            p = p[1:-1].strip()
    p = _ENUM.sub("", p)
    p = _LABEL.sub("", p)
    p = re.sub(r"\b[0-9A-Fa-f]{6}\b\.?", "", p)        # stray hex color codes the LLM leaks
    p = p.replace("\\n", " ").replace("\n", " ")
    p = re.sub(r"\s+", " ", p).strip()
    p = p.strip(' "\'`')
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=450)
    ap.add_argument("--field", default="prompt")
    args = ap.parse_args()

    seen, kept = set(), 0
    stats = Counter()
    with open(args.inp, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            p = clean_text(str(rec.get(args.field, "")))
            n = len(p)
            if n < args.min_chars or n > args.max_chars:
                stats["length"] += 1; continue
            if REFUSAL.search(p):
                stats["refusal"] += 1; continue
            if p.count(",") > n / 6:
                stats["tag_soup"] += 1; continue
            key = p.lower()
            if key in seen:
                stats["dup"] += 1; continue
            seen.add(key)
            rec[args.field] = p
            o.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1; stats["ok"] += 1
    print(f"kept {kept} -> {args.out} | drops: {dict(stats)}", file=sys.stderr)

if __name__ == "__main__":
    main()
