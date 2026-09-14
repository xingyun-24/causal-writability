import torch, os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, model_validator

from sshv2.wan._internal.model_loader_compat import (
    CompatibilityModelConfig as CustomModelConfig,
)


# MARK: Wan Config

class WanConfig(BaseModel):
    dit: 'WanDiTConfig'
    vae: 'WanVAEConfig | None'
    num_condition_frames: int = 0
    type: Literal["default", "autoregressive"] = "default"
    block_size: int = 1
    absorb_remainder: bool = True

    @model_validator(mode="after")
    def validate_pipeline_type(self):
        if self.dit.type == "block" and self.type != "autoregressive":
            raise ValueError(
                "Expected `WanConfig.type` to be 'autoregressive' when "
                f"`WanConfig.dit.type` is 'block', but got {self.type!r}."
            )
        return self

    @property
    def pipe(self):
        if self.type == "default":
            from sshv2.wan._internal.sampler_compat import WanVideoPipeline as cls
        elif self.type == "autoregressive":
            if self.dit.type == "block":
                raise ValueError("Block Wan was removed because it was unused and incomplete")
            else:
                raise ValueError("Causal Wan was removed because it was unused and incomplete")
        return cls

    @property
    def pipe_kwargs(self) -> dict:
        if self.dit.type == "block":
            return {"block_size": self.block_size, "absorb_remainder": self.absorb_remainder}
        return {}


# MARK: DiT Config

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
    has_text_input: bool
    has_image_input: bool
    num_inference_steps: int = 50
    spatial_pos_emb_size: tuple[int, int] = (128, 128)
    spatial_pos_emb_mode: Literal["rope", "learned", "absolute"] = "rope"
    type: Literal["bidirectional", "causal", "block"] = "bidirectional"
    ckpt_file: Path | None = None
    avg_ckpt_file: Path | None = None
    mechanism_adapter: Literal["none", "k_routing", "late_response", "combined", "shuffled_motion", "generic_response", "forced_canonical", "forced_global", "forced_shuffled", "forced_zero"] = "none"
    mechanism_rank: int = 32
    trajectory_renderer: bool = False
    # Number of explicitly tracked future trajectories supplied to the
    # renderer.  The original oracle-wall renderer uses one channel; the
    # multi-interaction renderer uses one fixed channel per interaction.
    trajectory_channels: int = 1
    # R1 supplies a separately encoded static prefix.  Do not silently alter
    # either of its two causal VAE slices inside the denoiser.
    renderer_context_exact: bool = False
    renderer_static_context: bool = False
    latent_canonical_mode: Literal["none", "full", "steered", "local_residual", "local_forced"] = "none"
    chart_adapter_ckpt: Path | None = None
    local_canonical_roi: tuple[float, float, float, float] = (0.18, 0.84, 0.39, 1.0)
    local_canonical_size: tuple[int, int] = (6, 6)
    local_canonical_layers: tuple[int, ...] = (7, 14, 22, 29)
    # Frozen Wan encoder-conv1 prefix condition.  It is strictly a visible
    # five-frame local crop; no simulator future state or output label enters.
    vae_local_conditioning: Literal["none", "forced_kv"] = "none"
    vae_local_condition_variant: Literal["tight_p1", "medium_p2"] = "tight_p1"

    @property
    def model(self):
        if self.type == "bidirectional":
            from sshv2.wan._internal.model_compat import WanModel as cls
        elif self.type == "causal":
            raise ValueError("Causal Wan was removed because it was unused and incomplete")
        elif self.type == "block":
            raise ValueError("Block Wan was removed because it was unused and incomplete")
        return cls

    @classmethod
    def from_default(cls, config: Literal['U2V-768d-30l', 'U2V-1536d-30l']):
        config_dict = cls._get_default_dict(config)
        return cls(**config_dict)
    
    @staticmethod
    def _get_default_dict(config: Literal['U2V-768d-30l', 'U2V-1536d-30l']):
        if config == 'U2V-768d-30l':
            return U2V_768d_30l
        elif config == 'U2V-1536d-30l':
            return U2V_1536d_30l
    
    def get_model_config(
        self,
        ckpt_file: str | os.PathLike | Path = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None
    ) -> CustomModelConfig:
        exclude_fields = {"type", "num_inference_steps", "ckpt_file", "avg_ckpt_file", "chart_adapter_ckpt"}
        return CustomModelConfig(
            type=self.model,
            name="wan_video_dit",
            kwargs=self.model_dump(exclude=exclude_fields),
            file_path=(ckpt_file or self.ckpt_file),
            device=device,
            torch_dtype=torch_dtype
        )


# MARK: VAE Config

class WanVAEConfig(BaseModel):
    z_dim: int
    queued: bool = False
    identity: bool = False
    ckpt_file: Path | None = None

    @classmethod
    def from_default(cls, config: Literal['default']):
        config_dict = cls._get_default_dict(config)
        return cls(**config_dict)
    
    @staticmethod
    def _get_default_dict(config: Literal['default']):
        if config == 'default':
            return VAE

    def get_model_config(
        self,
        ckpt_file: str | os.PathLike | Path = None,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
        identity: bool | None = None
    ) -> CustomModelConfig:
        if self.identity if identity is None else identity:
            from sshv2.wan._internal.vae import IdentityWanVideoVAE
            return CustomModelConfig(
                type=IdentityWanVideoVAE,
                name="wan_video_vae",
                kwargs={"z_dim": self.z_dim},
                device=device,
                torch_dtype=torch_dtype
            )
        from sshv2.wan._internal.vae import WanVideoVAE as cls
        return CustomModelConfig(
            type=cls,
            name="wan_video_vae",
            kwargs=self.model_dump(exclude=["identity", "ckpt_file"]),
            file_path=(ckpt_file or self.ckpt_file),
            device=device,
            torch_dtype=torch_dtype
        )
    
    @staticmethod
    def get_identity_model_config(z_dim: int = 3):
        from sshv2.wan._internal.vae import IdentityWanVideoVAE
        return CustomModelConfig(
            type=IdentityWanVideoVAE,
            name="wan_video_vae",
            kwargs={"z_dim": z_dim},
        )


# MARK: Defaults

U2V_768d_30l = {
    "dim": 768,
    "in_dim": 16,
    "ffn_dim": 3072,
    "out_dim": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 6,
    "num_layers": 30,
    "has_text_input": False,
    "has_image_input": False
}

U2V_1536d_30l = {
    "dim": 1536,
    "in_dim": 16,
    "ffn_dim": 6144,
    "out_dim": 16,
    "text_dim": 4096,
    "freq_dim": 256,
    "eps": 1e-6,
    "patch_size": (1, 2, 2),
    "num_heads": 12,
    "num_layers": 30,
    "has_text_input": False,
    "has_image_input": False
}

VAE = {
    "z_dim": 16,
    "queued": False
}
