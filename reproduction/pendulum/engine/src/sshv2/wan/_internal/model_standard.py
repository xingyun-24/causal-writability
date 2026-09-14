"""No-text Wan-like DiT used by the standard compatibility profile."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from diffsynth.models.wan_video_dit import (
    CrossAttention,
    GateModule,
    Head,
    MLP,
    SelfAttention,
    SimpleAdapter,
    WanModelStateDictConverter,
    modulate,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_text_input: bool,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.has_text_input = has_text_input
        self.has_image_input = has_image_input
        self.self_attn = SelfAttention(dim, num_heads, eps)
        if has_text_input or has_image_input:
            self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)
        x = self.gate(x, gate_msa, self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa), freqs))
        if self.has_text_input or self.has_image_input:
            x = x + self.cross_attn(self.norm3(x), context)
        x = self.gate(x, gate_mlp, self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class WanModel(nn.Module):
    """Compact Wan-like video DiT with optional text/image paths disabled here."""

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_text_input: bool,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_text_input = has_text_input
        self.has_image_input = has_image_input
        self.patch_size = tuple(patch_size)
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        if has_text_input:
            self.text_embedding = nn.Sequential(
                nn.Linear(text_dim, dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(dim, dim),
            )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [DiTBlock(has_text_input, has_image_input, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]
        )
        self.head = Head(dim, out_dim, self.patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(dim // num_heads)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.control_adapter = (
            SimpleAdapter(in_dim_control_adapter, dim, kernel_size=self.patch_size[1:], stride=self.patch_size[1:])
            if add_control_adapter
            else None
        )

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]):
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        if self.has_text_input:
            if context is None:
                raise ValueError("Text-enabled model requires context")
            context = self.text_embedding(context)
        else:
            context = x.new_zeros((x.shape[0], 1, self.dim))

        x = self.patchify(x)
        f, h, w = x.shape[2:]
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = torch.cat(
            [
                self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1).to(x.device)
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)
        x = self.head(x, t)
        return self.unpatchify(x, (f, h, w))

    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
