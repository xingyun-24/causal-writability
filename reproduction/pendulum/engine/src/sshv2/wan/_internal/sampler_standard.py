"""Minimal no-text pipeline for the shared Wan implementation."""

from __future__ import annotations

from typing import Optional, Union

import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from typing_extensions import override

from diffsynth.models.model_manager import ModelManager
from diffsynth.pipelines.wan_video_new import (
    ModelConfig,
    PipelineUnit,
    WanVideoPipeline as BaseWanVideoPipeline,
    sinusoidal_embedding_1d,
)
from sshv2.wan._internal.model_standard import WanModel
from sshv2.wan._internal.vae import IdentityWanVideoVAE
from sshv2.wan._internal.model_loader_standard import CustomModelConfig


DEFAULT_NUM_INFERENCE_STEPS = 50


class WanVideoPipeline(BaseWanVideoPipeline):
    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_path=None,
        default_num_inference_steps: int | None = None,
    ):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_InputVideoEmbedder(),
        ]
        self.post_units = []
        self.pre_encoded = False
        self.model_fn = model_fn_wan_video
        self.in_iteration_models = ("dit",)
        self.default_num_inference_steps = default_num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS

    def default_num_inference_steps_(self, value: int | None):
        self.default_num_inference_steps = value or DEFAULT_NUM_INFERENCE_STEPS

    def pre_encoded_(self, mode: bool = True):
        self.pre_encoded = mode

    def training_loss(self, **inputs):
        max_id = int(inputs.get("max_timestep_boundary", 1.0) * self.scheduler.num_train_timesteps)
        min_id = int(inputs.get("min_timestep_boundary", 0.0) * self.scheduler.num_train_timesteps)
        fixed_id = inputs.get("training_timestep_id")
        if fixed_id is None:
            timestep_id = torch.randint(min_id, max_id, (1,))
        else:
            timestep_id = torch.as_tensor(fixed_id, dtype=torch.long).reshape(1)
            if not min_id <= int(timestep_id.item()) < max_id:
                raise ValueError(f"training_timestep_id={int(timestep_id.item())} is outside [{min_id}, {max_id})")
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        clean = inputs["input_latents"].to(device=self.device, dtype=self.torch_dtype)
        noise = inputs["noise"].to(device=self.device, dtype=self.torch_dtype)
        latents = self.scheduler.add_noise(clean, noise, timestep)
        target = self.scheduler.training_target(clean, noise, timestep)

        num_condition_frames = int(inputs["num_condition_frames"])
        latents[:, :, :num_condition_frames] = clean[:, :, :num_condition_frames]
        noise_pred = self.model_fn(dit=inputs["dit"], latents=latents, timestep=timestep)

        noise_pred = noise_pred[:, :, num_condition_frames:]
        target = target[:, :, num_condition_frames:]
        if noise_pred.numel() == 0:
            raise ValueError("num_condition_frames leaves no future latent frames for the loss")
        loss = torch.nn.functional.mse_loss(noise_pred.float(), target.float())
        return loss * self.scheduler.training_weight(timestep)

    @override
    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        first = video[0]
        if isinstance(first, torch.Tensor):
            out = [video] if isinstance(video, torch.Tensor) else video
        else:
            out = []
            for item in ([video] if isinstance(first, Image.Image) else video):
                out.append(super().preprocess_video(item, torch_dtype, device, pattern, min_value, max_value))
        out = torch.cat(out, dim=pattern.index("B") // 2)
        return out.to(device=device or self.device, dtype=torch_dtype or self.torch_dtype)

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] | None = None,
        tokenizer_config: ModelConfig | None = None,
        use_usp: bool = False,
    ):
        if tokenizer_config is not None:
            raise ValueError("The no-text Wan pipeline does not accept a tokenizer config")
        if use_usp:
            raise ValueError("USP is not used by the no-text Wan pipeline")

        pipe = cls(device=device, torch_dtype=torch_dtype)
        manager = ModelManager(torch_dtype=torch_dtype, device=device)
        for config in model_configs or []:
            if isinstance(config, CustomModelConfig):
                config.load_model(manager)
            else:
                config.download_if_necessary(use_usp=False)
                manager.load_model(
                    config.path,
                    device=config.offload_device or device,
                    torch_dtype=config.offload_dtype or torch_dtype,
                )
        pipe.dit = manager.fetch_model("wan_video_dit")
        pipe.dit2 = None
        pipe.vae = manager.fetch_model("wan_video_vae")
        pipe.text_encoder = None
        pipe.image_encoder = None
        if pipe.dit is None or pipe.vae is None:
            raise RuntimeError("Both wan_video_dit and wan_video_vae must be available")
        pipe.height_division_factor = pipe.vae.upsampling_factor * 2
        pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = "",
        negative_prompt: Optional[str] = "",
        cfg_scale: float = 1.0,
        height: int = 128,
        width: int = 128,
        num_frames: int = 69,
        num_condition_frames: int = 0,
        condition_frames: list[Image.Image] | torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        denoising_strength: float = 1.0,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        num_samples: int = 1,
        return_as_tensor: bool = False,
        return_latents: bool = False,
        progress_bar_cmd=tqdm,
        seed: int | None = None,
    ):
        del prompt, negative_prompt
        if cfg_scale != 1.0:
            raise ValueError("The no-text Wan model requires cfg_scale=1.0")
        if num_condition_frames <= 0:
            raise ValueError("num_condition_frames must be positive")
        if condition_frames is None:
            raise ValueError("condition_frames are required for continuation generation")

        self.train(False)
        scheduler_was_training = bool(getattr(self.scheduler, "training", False))
        scheduler_shift = getattr(self.scheduler, "shift", None)
        steps = num_inference_steps or self.default_num_inference_steps
        self.scheduler.set_timesteps(steps, denoising_strength=denoising_strength, shift=sigma_shift)

        shared = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_samples": num_samples,
            "seed": seed,
            "rand_device": self.device,
            "input_video": None,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }
        positive, negative = {}, {}
        for unit in self.units:
            shared, positive, negative = self.unit_runner(unit, self, shared, positive, negative)

        self.load_models_to_device(self.in_iteration_models)
        self.load_models_to_device(("vae",))
        condition_tensor = self.preprocess_video(condition_frames)
        condition_latents = (
            condition_tensor
            if self.pre_encoded
            else self.vae.encode(
                condition_tensor,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        )
        condition_latents = condition_latents[:, :, :num_condition_frames].to(
            dtype=self.torch_dtype, device=self.device
        )
        if condition_latents.shape[2] != num_condition_frames:
            raise ValueError(
                f"Condition has {condition_latents.shape[2]} latent frames; expected {num_condition_frames}"
            )

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            shared["latents"][:, :, :num_condition_frames] = condition_latents
            noise_pred = self.model_fn(dit=self.dit, latents=shared["latents"], timestep=timestep)
            shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                shared["latents"],
            )
            shared["latents"][:, :, :num_condition_frames] = condition_latents

        final_latents = shared["latents"].detach().clone() if return_latents else None

        self.load_models_to_device(("vae",))
        video = self.vae.decode(
            shared["latents"],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if not return_as_tensor:
            video = [self.vae_output_to_video(item.unsqueeze(0)) for item in video]
        self.load_models_to_device([])

        if scheduler_was_training:
            kwargs = {"training": True}
            if scheduler_shift is not None:
                kwargs["shift"] = scheduler_shift
            self.scheduler.set_timesteps(self.scheduler.num_train_timesteps, **kwargs)
        return (video, final_latents) if return_latents else video


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        if not isinstance(pipe.vae, IdentityWanVideoVAE):
            height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "num_samples", "seed", "rand_device"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, num_samples, seed, rand_device):
        length = num_frames if isinstance(pipe.vae, IdentityWanVideoVAE) else (num_frames - 1) // 4 + 1
        shape = (
            num_samples or 1,
            pipe.vae.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        return {"noise": pipe.generate_noise(shape, seed=seed, rand_device=rand_device)}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",),
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(("vae",))
        tensor = pipe.preprocess_video(input_video)
        input_latents = (
            tensor
            if pipe.pre_encoded
            else pipe.vae.encode(tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        )
        input_latents = input_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        return {"latents": pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])}


def has_text_input(dit):
    return not hasattr(dit, "has_text_input") or dit.has_text_input


def get_spatial_positional_inputs(
    dit,
    x: torch.Tensor,
    f: int,
    h: int,
    w: int,
):
    """Return the standard 3-D RoPE inputs used by shared Wan variants."""
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)
    return freqs, None


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    **_: object,
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = latents.new_zeros((latents.shape[0], 1, dit.dim))

    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)
    for block in dit.blocks:
        x = block(x, context, t_mod, freqs)
    x = dit.head(x, t)
    return dit.unpatchify(x, (f, h, w))
