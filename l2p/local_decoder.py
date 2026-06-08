"""
Local pixel decoder for L2P (Latent->Pixel) on Qwen-Image.

A small U-Net ("MicroDiffusionModel") that takes the *noisy RGB image* and the
DiT's per-patch feature map (injected at the bottleneck) and predicts the
flow-matching velocity directly in pixel space — replacing the VAE decoder.

Ported from the Z-Image L2P implementation (DiP / L2P papers). The encoder
pools 4× by stride-2 (total 16×), which matches the L2P pixel patch size of 16:
a 16×16 pixel patch == one DiT token, so the bottleneck spatial grid equals the
DiT feature-map grid (H/16, W/16). si_t_hidden_size = transformer inner dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


def _gc(fn, use_ckpt, use_offload, *args):
    if use_ckpt and torch.is_grad_enabled():
        if use_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return ckpt.checkpoint(fn, *args, use_reentrant=False)
        return ckpt.checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)

class MicroDiffusionModel(nn.Module):
    def __init__(self, in_channels: int, si_t_hidden_size: int):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 64, 3, padding=1), nn.SiLU())
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.SiLU())
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.SiLU())
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.SiLU())
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.bottleneck = nn.Sequential(nn.Conv2d(512 + si_t_hidden_size, 512, 1), nn.SiLU())

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 512, 3, padding=1))
        self.dec4 = nn.Sequential(nn.Conv2d(512 + 512, 256, 3, padding=1), nn.SiLU())
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 256, 3, padding=1))
        self.dec3 = nn.Sequential(nn.Conv2d(256 + 256, 128, 3, padding=1), nn.SiLU())
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, padding=1))
        self.dec2 = nn.Sequential(nn.Conv2d(128 + 128, 64, 3, padding=1), nn.SiLU())
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(64, 64, 3, padding=1))
        self.dec1 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(64, in_channels, 1)
        # zero-init the output conv so the initial velocity prediction is 0:
        # MSE then starts at E[(noise-x0)^2] ~ O(1) instead of exploding from a
        # random head fed the large-magnitude DiT residual stream.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, c, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False):
        def _enc1(x_in):
            e = self.enc1(x_in)
            return e, self.pool1(e)

        def _enc2(p):
            e = self.enc2(p)
            return e, self.pool2(e)

        def _enc3(p):
            e = self.enc3(p)
            return e, self.pool3(e)

        def _enc4_bottleneck(p, c_in):
            e = self.enc4(p)
            p4 = self.pool4(e)
            if c_in.shape[-2:] != p4.shape[-2:]:
                c_in = F.interpolate(c_in, size=p4.shape[-2:], mode="nearest")
            b = self.bottleneck(torch.cat([p4, c_in], dim=1))
            return e, b

        def _dec4(b, e4):
            d = self.up4(b)
            return self.dec4(torch.cat([d, e4], dim=1))

        def _dec3(d_in, e3):
            d = self.up3(d_in)
            return self.dec3(torch.cat([d, e3], dim=1))

        def _dec2(d_in, e2):
            d = self.up2(d_in)
            return self.dec2(torch.cat([d, e2], dim=1))

        def _dec1_out(d_in, e1):
            d = self.up1(d_in)
            d = self.dec1(torch.cat([d, e1], dim=1))
            return self.out_conv(d)

        gc = use_gradient_checkpointing
        off = use_gradient_checkpointing_offload
        e1, p1 = _gc(_enc1, gc, off, x)
        e2, p2 = _gc(_enc2, gc, off, p1)
        e3, p3 = _gc(_enc3, gc, off, p2)
        e4, b = _gc(_enc4_bottleneck, gc, off, p3, c)
        d4 = _gc(_dec4, gc, off, b, e4)
        d3 = _gc(_dec3, gc, off, d4, e3)
        d2 = _gc(_dec2, gc, off, d3, e2)
        out = _gc(_dec1_out, gc, off, d2, e1)
        return out
