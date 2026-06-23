import csv, json, math, os
import torch
from PIL import Image
import numpy as np

def target_hw(h, w, base):
    """Scale (h,w) to ~base*base AREA, preserve aspect ratio, round each side to a multiple
    of 16 (the pixel patch size), min 16. base<=0 -> unchanged (native resolution).
    Area-normalizing (not shorter-side) keeps memory uniform across aspect-ratio buckets."""
    if not base or base <= 0:
        return h, w
    s = base / math.sqrt(h * w)
    nh = max(16, int(round(h * s / 16)) * 16)
    nw = max(16, int(round(w * s / 16)) * 16)
    return nh, nw

class OverfitDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, text_cache=None, repeat=1, limit=None, resize_base=0):
        self.data_dir = data_dir
        self.resize_base = resize_base
        self.text_cache = text_cache or os.path.join(data_dir, "text_cache")
        self.rows = list(csv.DictReader(open(os.path.join(data_dir, "metadata.csv"), encoding="utf-8")))
        if limit:
            self.rows = self.rows[:limit]
        self.index = json.load(open(os.path.join(self.text_cache, "index.json")))
        self.repeat = repeat

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, i):
        r = self.rows[i % len(self.rows)]
        img = Image.open(os.path.join(self.data_dir, r["file_name"])).convert("RGB")
        if self.resize_base:
            nh, nw = target_hw(img.height, img.width, self.resize_base)
            if (nh, nw) != (img.height, img.width):
                img = img.resize((nw, nh), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)  # [3,H,W] 0..255
        x = x / 127.5 - 1.0                                                        # -> [-1, 1]
        stem = self.index[r["file_name"]]
        emb = torch.load(os.path.join(self.text_cache, f"{stem}.pt"), weights_only=True)["prompt_embeds"]
        return {"image": x, "prompt_embeds": emb, "file_name": r["file_name"], "text": r["text"]}

    def image_sizes(self):
        """(H, W) per effective index via cheap PIL header reads (no decode), cached.
        Used by the resolution-bucketed batch sampler so >1 batches only stack same-size images."""
        if getattr(self, "_sizes", None) is None:
            per_row = []
            for r in self.rows:
                with Image.open(os.path.join(self.data_dir, r["file_name"])) as im:
                    w, h = im.size
                per_row.append(target_hw(h, w, self.resize_base))
            n = len(self.rows)
            self._sizes = [per_row[i % n] for i in range(len(self))]
        return self._sizes

