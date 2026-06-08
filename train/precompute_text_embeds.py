#!/usr/bin/env python3
import argparse, csv, json, os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer

PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_START_IDX = 34
TOKENIZER_MAX_LENGTH = 1024

def extract_masked(hidden, mask):
    return [hidden[i, mask[i].bool()] for i in range(hidden.shape[0])]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image-2512")
    ap.add_argument("--data_dir", required=True, help="dir with metadata.csv (from curate_overfit.py)")
    ap.add_argument("--out", default=None, help="cache dir (default: <data_dir>/text_cache)")
    ap.add_argument("--max_sequence_length", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = args.out or os.path.join(args.data_dir, "text_cache")
    os.makedirs(out, exist_ok=True)

    template = PROMPT_TEMPLATE
    drop_idx = PROMPT_TEMPLATE_START_IDX
    tok_max = TOKENIZER_MAX_LENGTH

    src = args.model if os.path.isdir(args.model) else args.model
    print(f"[load] text encoder + tokenizer from {src}")
    if os.path.isdir(src):
        te_path, tk_path = os.path.join(src, "text_encoder"), os.path.join(src, "tokenizer")
    else:
        te_path = tk_path = src
    if os.path.isdir(src):
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(te_path, dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(tk_path)
    else:
        te = Qwen2_5_VLForConditionalGeneration.from_pretrained(src, subfolder="text_encoder", dtype=torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(src, subfolder="tokenizer")
    te = te.to(args.device).eval()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "metadata.csv"), encoding="utf-8")))
    print(f"[data] {len(rows)} prompts")

    index = {}
    for i, r in enumerate(rows):
        prompt = r["text"]
        txt = template.format(prompt)
        ti = tok([txt], max_length=tok_max + drop_idx, padding=True, truncation=True, return_tensors="pt").to(args.device)
        hs = te(input_ids=ti.input_ids, attention_mask=ti.attention_mask, output_hidden_states=True).hidden_states[-1]
        emb = extract_masked(hs, ti.attention_mask)[0][drop_idx:]      # [T, 3584]
        emb = emb[:args.max_sequence_length].to(torch.float16).cpu()
        stem = os.path.splitext(os.path.basename(r["file_name"]))[0]
        torch.save({"prompt_embeds": emb}, os.path.join(out, f"{stem}.pt"))
        index[r["file_name"]] = stem
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] last T={emb.shape[0]}")

    json.dump(index, open(os.path.join(out, "index.json"), "w"))
    print(f"[done] cached {len(index)} -> {out}")

if __name__ == "__main__":
    main()
