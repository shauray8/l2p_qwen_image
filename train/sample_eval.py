import math
import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "l2p"))
from qwen_image_l2p import PIXEL_PATCH  # noqa: E402

_LPIPS = None

def qwen_sigma_schedule(num_steps, image_seq_len, shift_terminal=0.02):
    m = (0.9 - 0.5) / (8192 - 256)
    mu = image_seq_len * m + (0.5 - m * 256)
    s = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]
    s = math.exp(mu) / (math.exp(mu) + (1.0 / s.clamp(1e-6, 1) - 1.0))
    one_minus = 1 - s
    s = 1 - one_minus / (one_minus[-1] / (1 - shift_terminal))
    return torch.cat([s, s.new_zeros(1)])  # append final 0

@torch.no_grad()
def sample(model, prompt_embeds, height, width, steps, device, seed=0, dtype=torch.bfloat16,
           uncond_embeds=None, cfg_scale=1.0):
    model_was_training = model.training
    model.eval()
    image_seq_len = (height // PIXEL_PATCH) * (width // PIXEL_PATCH)
    sigmas = qwen_sigma_schedule(steps, image_seq_len).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, 3, height, width, generator=g, device=device, dtype=torch.float32)
    emb = prompt_embeds.unsqueeze(0).to(device, dtype)
    eu = uncond_embeds.unsqueeze(0).to(device, dtype) if uncond_embeds is not None else None
    do_cfg = eu is not None and cfg_scale != 1.0
    for i in range(steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        v = model(noisy_image=x, timestep=sigma.view(1), prompt_embeds=emb,
                  prompt_embeds_mask=None, use_gradient_checkpointing=False).float()
        if do_cfg:
            vu = model(noisy_image=x, timestep=sigma.view(1), prompt_embeds=eu,
                       prompt_embeds_mask=None, use_gradient_checkpointing=False).float()
            v = vu + cfg_scale * (v - vu)
        x = x + (sigma_next - sigma) * v
    if model_was_training:
        model.train()
    img = ((x[0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return img  # HxWx3 uint8

def _psnr(a, b):  # a,b uint8 HxWx3
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 100.0 if mse == 0 else 20 * math.log10(255.0) - 10 * math.log10(mse)

def _lpips(a, b, device):
    global _LPIPS
    try:
        if _LPIPS is None:
            import lpips
            _LPIPS = lpips.LPIPS(net="alex").to(device).eval()
        ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
        tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().to(device) / 127.5 - 1
        with torch.no_grad():
            return float(_LPIPS(ta, tb).item())
    except Exception:
        return None

@torch.no_grad()
def recon_metrics(model, eval_items, steps, device, seed=0, uncond_embeds=None, cfg_scale=1.0):
    psnrs, lpips_vals, pairs = [], [], []
    for it in eval_items:
        H, W = it["image_np"].shape[:2]
        gen = sample(model, it["prompt_embeds"], H, W, steps, device, seed=seed,
                     uncond_embeds=uncond_embeds, cfg_scale=cfg_scale)
        psnrs.append(_psnr(gen, it["image_np"]))
        lv = _lpips(gen, it["image_np"], device)
        if lv is not None:
            lpips_vals.append(lv)
        pairs.append((gen, it["image_np"], it["file_name"]))
    return (float(np.mean(psnrs)),
            float(np.mean(lpips_vals)) if lpips_vals else None,
            pairs)
