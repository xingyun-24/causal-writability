"""Checkpoint-compatible raw-video pipeline in the shared Wan package."""

import torch
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from typing import Optional, Union
from typing_extensions import override

from diffsynth.pipelines.wan_video_new import *
from sshv2.wan._internal.model_loader_compat import CompatibilityModelConfig as CustomModelConfig
from sshv2.wan._internal.vae import IdentityWanVideoVAE
from diffsynth.pipelines.wan_video_new import WanVideoPipeline as BaseWanVideoPipeline


# MARK: Wan

DEFAULT_NUM_INFERENCE_STEPS = 50

class WanVideoPipeline(BaseWanVideoPipeline):
    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, default_num_inference_steps: int | None = None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_InputVideoEmbedder()
        ]
        self.post_units = []
        self.pre_encoded = False
        self.model_fn = model_fn_wan_video
        self.in_iteration_models = ("dit",)
        self.default_num_inference_steps = default_num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS

    def default_num_inference_steps_(self, n):
        self.default_num_inference_steps = n or DEFAULT_NUM_INFERENCE_STEPS

    def pre_encoded_(self, mode: bool = True):
        self.pre_encoded = mode
        
    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        clean = inputs["input_latents"]
        # The full latent-canonical branch operates after VAE encoding.  Noise
        # is sampled directly in this canonical grid, so no future state is
        # ever transformed or supplied as a condition.
        mode = getattr(inputs["dit"], "latent_canonical_mode", "none")
        if mode in ("full", "steered"):
            raise ValueError(
                "Removed latent-canonical prototypes are not supported"
            )
        inputs["latents"] = self.scheduler.add_noise(clean, inputs["noise"], timestep)
        training_target = self.scheduler.training_target(clean, inputs["noise"], timestep)

        num_condition_frames = inputs["num_condition_frames"]
        condition_latents = inputs.get("renderer_context", clean)
        condition_latents = condition_latents.to(device=self.device, dtype=self.torch_dtype)
        inputs["latents"][:, :, :num_condition_frames, ...] = condition_latents[:, :, :num_condition_frames, ...]
        
        noise_pred = self.model_fn(**inputs, timestep=timestep)

        noise_pred = noise_pred[:, :, num_condition_frames:, ...]
        training_target = training_target[:, :, num_condition_frames:, ...]
        
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    @override
    def preprocess_video(self, video, torch_dtype=None, device=None, pattern="B C T H W", min_value=-1, max_value=1):
        first = video[0]
        if isinstance(first, torch.Tensor):
            out = [video] if isinstance(video, torch.Tensor) else video
        else: # list of PIL.Image
            out = []
            for vid in [video] if isinstance(first, Image.Image) else video:
                out.append(super().preprocess_video(vid, torch_dtype, device, pattern, min_value, max_value))
        out = torch.cat(out, dim=pattern.index("B") // 2)
        out = out.to(device=device or self.device, dtype=torch_dtype or self.torch_dtype)
        return out

    @classmethod
    def from_pretrained(
        cls,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = None,
        use_usp=False,
    ):
        pipe = cls(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()

        model_manager = ModelManager(torch_dtype=torch_dtype, device=device)
        for model_config in model_configs:
            if isinstance(model_config, CustomModelConfig):
                model_config.load_model(model_manager)
                continue
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        pipe.dit, pipe.dit2 = dit if isinstance(dit, list) else (dit, None)
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2
        
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary(use_usp=use_usp)
            pipe.prompter.fetch_models(pipe.text_encoder)
            pipe.prompter.fetch_tokenizer(tokenizer_config.path)
        
        if use_usp: pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[str] = "",
        negative_prompt: Optional[str] = "",
        cfg_scale: float = 7.0,
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        num_condition_frames: int = 0,
        condition_frames: list[Image.Image] | torch.Tensor | None = None,
        num_inference_steps: int | None = None,
        denoising_strength: float = 1.0,
        sigma_shift: float = 5.0,
        tiled: bool | None = True,
        tile_size: tuple[int, int] | None = (30, 52),
        tile_stride: tuple[int, int] | None = (15, 26),
        num_samples: int = 1,
        return_as_tensor: bool = False,
        progress_bar_cmd = tqdm,
        seed: int | None = None,
        motion_tube: torch.Tensor | None = None,
        canonical_motion: torch.Tensor | None = None,
        trajectory_maps: torch.Tensor | None = None,
        wall_geometry: torch.Tensor | None = None,
        vae_local_condition: torch.Tensor | None = None,
    ):
        self.train(False)

        # A training module and its validation sampler share this scheduler.
        # Preserve the full 1,000-step, training-weighted schedule so a
        # validation rollout cannot leave the next optimizer step using its
        # short inference schedule.
        scheduler_was_training = bool(getattr(self.scheduler, "training", False))
        scheduler_shift = getattr(self.scheduler, "shift", None)

        if num_inference_steps is None:
            num_inference_steps = getattr(self, "default_num_inference_steps", 50)
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}

        inputs_shared = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_condition_frames": num_condition_frames,
            "condition_frames": condition_frames,
            "denoising_strength": denoising_strength,
            "cfg_scale": cfg_scale,
            "sigma_shift": sigma_shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "num_samples": num_samples,
            "seed": seed,
            "rand_device": self.device
        }
        if motion_tube is not None:
            inputs_shared["motion_tube"] = motion_tube.to(device=self.device)
        if canonical_motion is not None:
            inputs_shared["canonical_motion"] = canonical_motion.to(device=self.device)
        if trajectory_maps is not None:
            inputs_shared["trajectory_maps"] = trajectory_maps.to(device=self.device)
        if wall_geometry is not None:
            inputs_shared["wall_geometry"] = wall_geometry.to(device=self.device)
        if vae_local_condition is not None:
            inputs_shared["vae_local_condition"] = vae_local_condition.to(device=self.device, dtype=self.torch_dtype)

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        if condition_frames is not None:
            self.load_models_to_device(["vae"])
            condition_frames = self.preprocess_video(condition_frames)
            condition_latents = condition_frames if self.pre_encoded else \
                self.vae.encode(condition_frames, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            condition_latents = condition_latents[:,:,:num_condition_frames].to(dtype=self.torch_dtype, device=self.device)
            if getattr(self.dit, "latent_canonical_mode", "none") in (
                "full",
                "steered",
            ):
                raise ValueError(
                    "Removed latent-canonical prototypes are not supported"
                )

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            if condition_frames is not None:
                inputs_shared["latents"][:,:,:num_condition_frames] = condition_latents
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if condition_frames is not None:
                inputs_shared["latents"][:,:,:num_condition_frames] = condition_latents
        
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        
        self.load_models_to_device(["vae"])
        output_latents = inputs_shared["latents"]
        if getattr(self.dit, "latent_canonical_mode", "none") in (
            "full",
            "steered",
        ):
            raise ValueError(
                "Removed latent-canonical prototypes are not supported"
            )
        video = self.vae.decode(output_latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if not return_as_tensor:
            video = [self.vae_output_to_video(vid.unsqueeze(0)) for vid in video]
        self.load_models_to_device([])

        if scheduler_was_training:
            restore_kwargs = {"training": True}
            if scheduler_shift is not None:
                restore_kwargs["shift"] = scheduler_shift
            self.scheduler.set_timesteps(self.scheduler.num_train_timesteps, **restore_kwargs)

        return video


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
        if isinstance(pipe.vae, IdentityWanVideoVAE):
            length = num_frames
        else:
            length = (num_frames - 1) // 4 + 1
        shape = (num_samples or 1, pipe.vae.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if has_text_input(pipe.dit) and pipe.text_encoder is not None and prompt is not None:
            prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        else:
            prompt_emb = None
        return {"context": prompt_emb}


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = input_video if pipe.pre_encoded else \
            pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        input_latents = input_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}


def has_text_input(dit):
    return not hasattr(dit, "has_text_input") or dit.has_text_input


def has_absolute_spatial_pos_emb(dit):
    return getattr(dit, "spatial_pos_emb_mode", "rope") != "rope"


def get_spatial_positional_inputs(dit, x: torch.Tensor, f: int, h: int, w: int):
    if has_absolute_spatial_pos_emb(dit):
        freqs = dit.freqs_temporal[:f].view(f, 1, 1, -1).expand(f, h, w, -1).reshape(f * h * w, 1, -1).to(x.device)
        h_idx = torch.arange(h, device=x.device)
        w_idx = torch.arange(w, device=x.device)
        spatial_emb = (
            torch.nn.functional.embedding(h_idx, dit.spatial_pos_emb_h).unsqueeze(1)
            + torch.nn.functional.embedding(w_idx, dit.spatial_pos_emb_w).unsqueeze(0)
        )
        spatial_emb = spatial_emb.unsqueeze(0).expand(f, -1, -1, -1).reshape(1, f * h * w, dit.dim).to(dtype=x.dtype)
        return freqs, spatial_emb

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    return freqs, None


def model_fn_wan_video(
    dit: WanModel,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    motion_tube: torch.Tensor = None,
    canonical_motion: torch.Tensor = None,
    trajectory_maps: torch.Tensor = None,
    wall_geometry: torch.Tensor = None,
    vae_local_condition: torch.Tensor = None,
    **kwargs
):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    x = latents
    if dit.renderer_static_context and not getattr(dit, "renderer_context_exact", False):
        x = x.clone(); x[:, :, :2] = x[:, :, 1:2]
    ctx_tokens = []

    if has_text_input(dit) and context is not None:
        text_ctx = dit.text_embedding(context)
        ctx_tokens.append(text_ctx)

        # TODO: Review CFG logic
        if x.shape[0] != text_ctx.shape[0]:
            x = torch.concat([x] * text_ctx.shape[0], dim=0)
        if timestep.shape[0] != text_ctx.shape[0]:
            timestep = torch.concat([timestep] * text_ctx.shape[0], dim=0)

    if len(ctx_tokens) > 0:
        context = torch.cat(ctx_tokens, dim=1)
    else:
        B = x.shape[0]
        context = x.new_zeros(B, 1, dit.dim)

    x = dit.patchify(x)
    if trajectory_maps is not None:
        if dit.trajectory_map_embed is None: raise ValueError("trajectory_maps supplied without trajectory_renderer")
        x = x + dit.trajectory_map_embed(trajectory_maps.to(device=x.device, dtype=x.dtype))

    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs, spatial_emb = get_spatial_positional_inputs(dit, x, f, h, w)
    if spatial_emb is not None:
        x = x + spatial_emb

    dit.mechanism_adapter.prepare(x, motion_tube, canonical_motion)
    if dit.vae_local is not None:
        if vae_local_condition is None:
            raise ValueError("forced VAE-local condition is enabled but no prefix crop was supplied")
        dit.vae_local.prepare(vae_local_condition.to(device=x.device, dtype=x.dtype))
    if dit.local_canonical is not None:
        if wall_geometry is None:
            raise ValueError("local latent canonicalization requires visible wall_geometry")
        dit.local_canonical.prepare(wall_geometry.to(device=x.device, dtype=torch.float32))
    for layer_id, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs, mechanism=dit.mechanism_adapter,
                  layer_id=layer_id, vae_local=dit.vae_local)
        if dit.local_canonical is not None:
            x = dit.local_canonical.apply(x, layer_id, t, (f, h, w))
            
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
