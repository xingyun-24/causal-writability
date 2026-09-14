"""Wan model configuration frozen for spring stage 1."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from pydantic import BaseModel



class WanConfig(BaseModel):
    dit: "WanDiTConfig"
    vae: "WanVAEConfig"
    num_condition_frames: int

    @property
    def pipe(self):
        from sshv2.diffsynth.pipelines.wan_video import WanVideoPipeline

        return WanVideoPipeline

    @property
    def pipe_kwargs(self) -> dict:
        return {}


class WanDiTConfig(BaseModel):
    dim: int
    in_dim: int
    ffn_dim: int
    out_dim: int
    text_dim: int
    freq_dim: int
    eps: float
    patch_size: tuple[int, int, int]
    num_heads: int
    num_layers: int
    has_text_input: bool = False
    has_image_input: bool = False
    num_inference_steps: int = 20
    type: str = "bidirectional"
    ckpt_file: Path | None = None
    avg_ckpt_file: Path | None = None

    def get_model_config(
        self,
        ckpt_file: str | os.PathLike | Path | None = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        if self.type != "bidirectional":
            raise ValueError("Spring stage 1 only supports the bidirectional DiT")
        from sshv2.diffsynth.models.wan_video_dit import WanModel
        from sshv2.diffsynth.utils.custom_config import CustomModelConfig

        return CustomModelConfig(
            type=WanModel,
            name="wan_video_dit",
            kwargs=self.model_dump(exclude={"type", "num_inference_steps", "ckpt_file", "avg_ckpt_file"}),
            file_path=ckpt_file or self.ckpt_file,
            device=device,
            torch_dtype=torch_dtype,
        )


class WanVAEConfig(BaseModel):
    z_dim: int = 16
    queued: bool = False
    ckpt_file: Path

    def get_model_config(
        self,
        ckpt_file: str | os.PathLike | Path | None = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ):
        from sshv2.diffsynth.models.wan_video_vae import WanVideoVAE
        from sshv2.diffsynth.utils.custom_config import CustomModelConfig

        return CustomModelConfig(
            type=WanVideoVAE,
            name="wan_video_vae",
            kwargs=self.model_dump(exclude={"ckpt_file"}),
            file_path=ckpt_file or self.ckpt_file,
            device=device,
            torch_dtype=torch_dtype,
        )
