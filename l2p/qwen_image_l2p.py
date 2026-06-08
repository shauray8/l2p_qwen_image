from typing import List, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from diffusers import QwenImageTransformer2DModel

try:
    from .local_decoder import MicroDiffusionModel
except ImportError:
    from local_decoder import MicroDiffusionModel

PIXEL_PATCH = 16
IN_CH = 3


def patchify_pixels(img: torch.Tensor, p: int = PIXEL_PATCH):
    B, C, H, W = img.shape
    assert H % p == 0 and W % p == 0, f"H,W must be divisible by {p}, got {H}x{W}"
    Ht, Wt = H // p, W // p
    x = img.view(B, C, Ht, p, Wt, p).permute(0, 2, 4, 1, 3, 5).reshape(B, Ht * Wt, C * p * p)
    return x, Ht, Wt


class QwenImageL2P(nn.Module):
    def __init__(
        self,
        dit: QwenImageTransformer2DModel,
        feature_use_norm_out: bool = True,
        pixel_patch: int = PIXEL_PATCH,
    ):
        super().__init__()
        self.dit = dit
        self.pixel_patch = pixel_patch
        self.feature_use_norm_out = feature_use_norm_out
        self.inner_dim = dit.inner_dim
        self.local_decoder = MicroDiffusionModel(in_channels=IN_CH, si_t_hidden_size=dit.inner_dim)

    @classmethod
    def from_config(cls, transformer_config: dict, **kwargs):
        cfg = dict(transformer_config)
        cfg["in_channels"] = IN_CH * PIXEL_PATCH * PIXEL_PATCH  # 768
        dit = QwenImageTransformer2DModel(**cfg)
        return cls(dit, **kwargs)

    def _run_backbone(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                      timestep, img_shapes, use_gradient_checkpointing, use_gradient_checkpointing_offload):
        dit = self.dit
        hidden_states = dit.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = dit.txt_norm(encoder_hidden_states)
        encoder_hidden_states = dit.txt_in(encoder_hidden_states)

        text_seq_len = encoder_hidden_states.shape[1]
        temb = dit.time_text_embed(timestep, hidden_states)
        image_rotary_emb = dit.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        block_kwargs = {}
        if encoder_hidden_states_mask is not None:
            B, img_len = hidden_states.shape[:2]
            image_mask = torch.ones((B, img_len), dtype=torch.bool, device=hidden_states.device)
            joint_mask = torch.cat([encoder_hidden_states_mask.bool(), image_mask], dim=1)[:, None, None, :]
            block_kwargs["attention_mask"] = joint_mask

        use_ckpt = use_gradient_checkpointing and torch.is_grad_enabled()
        for block in dit.transformer_blocks:
            if use_ckpt:
                def run(hs, ehs, tb, block=block):
                    return block(hidden_states=hs, encoder_hidden_states=ehs,
                                 encoder_hidden_states_mask=None, temb=tb,
                                 image_rotary_emb=image_rotary_emb,
                                 joint_attention_kwargs=block_kwargs, modulate_index=None)
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        encoder_hidden_states, hidden_states = ckpt.checkpoint(
                            run, hidden_states, encoder_hidden_states, temb, use_reentrant=False)
                else:
                    encoder_hidden_states, hidden_states = ckpt.checkpoint(
                        run, hidden_states, encoder_hidden_states, temb, use_reentrant=False)
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None, temb=temb,
                    image_rotary_emb=image_rotary_emb, joint_attention_kwargs=block_kwargs,
                    modulate_index=None)

        if self.feature_use_norm_out:
            hidden_states = dit.norm_out(hidden_states, temb)
        return hidden_states  # [B, img_len, dim]

    def forward(
        self,
        noisy_image: torch.Tensor,            # [B, 3, H, W]
        timestep: torch.Tensor,               # [B], flow-matching sigma in [0,1]
        prompt_embeds: torch.Tensor,          # [B, T, joint_attention_dim]
        prompt_embeds_mask: Optional[torch.Tensor] = None,  # [B, T]
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ) -> torch.Tensor:
        B, C, H, W = noisy_image.shape
        tokens, Ht, Wt = patchify_pixels(noisy_image, self.pixel_patch)
        tokens = tokens.to(self.dit.img_in.weight.dtype)
        img_shapes = [(1, Ht, Wt)] * B

        img_hidden = self._run_backbone(
            tokens, prompt_embeds, prompt_embeds_mask, timestep, img_shapes,
            use_gradient_checkpointing, use_gradient_checkpointing_offload)

        feat_map = img_hidden.view(B, Ht, Wt, self.inner_dim).permute(0, 3, 1, 2).contiguous()
        velocity = self.local_decoder(
            noisy_image.to(feat_map.dtype), feat_map,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        return velocity  # [B, 3, H, W] — flow-matching velocity (noise - x0)

