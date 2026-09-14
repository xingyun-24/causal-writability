"""Spring-only training module for the no-text Wan-like DiT."""

from __future__ import annotations

import torch

from sshv2.diffsynth.configs.wan_config import WanDiTConfig, WanVAEConfig
from sshv2.diffsynth.pipelines.wan_video import WanVideoPipeline
from sshv2.diffsynth.trainers.utils import DiffusionTrainingModule


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        dit_config: WanDiTConfig,
        vae_config: WanVAEConfig,
        no_encoding: bool,
        num_condition_frames: int,
        num_inference_steps: int | None = None,
        pipeline_type=WanVideoPipeline,
        pipeline_kwargs: dict | None = None,
    ):
        super().__init__()
        self.num_inference_steps = num_inference_steps or dit_config.num_inference_steps
        model_configs = [
            dit_config.get_model_config(torch_dtype=torch.bfloat16),
            vae_config.get_model_config(torch_dtype=torch.bfloat16),
        ]
        self.pipe = pipeline_type.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )
        for key, value in (pipeline_kwargs or {}).items():
            setattr(self.pipe, key, value)
        self.pipe.pre_encoded_(no_encoding)
        self.pipe.default_num_inference_steps_(self.num_inference_steps)

        self.switch_pipe_to_training_mode(
            pipe=self.pipe,
            trainable_models="dit",
            lora_base_model=None,
            lora_target_modules=None,
            lora_rank=None,
            lora_checkpoint=None,
            enable_fp8_training=False,
        )
        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs = []
        self.max_timestep_boundary = 1.0
        self.min_timestep_boundary = 0.0
        self.no_encoding = no_encoding
        self.num_condition_frames = num_condition_frames

    def get_shape(self, videos):
        first = videos[0]
        if isinstance(first, torch.Tensor):
            height, width, num_frames = first.shape[-2], first.shape[-1], first.shape[-3]
        else:
            height, width, num_frames = first[0].size[1], first[0].size[0], len(first)
        if self.no_encoding:
            height *= self.pipe.vae.upsampling_factor
            width *= self.pipe.vae.upsampling_factor
            num_frames = self.pipe.vae.to_num_frames(num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}

    def forward_preprocess(self, data):
        inputs = {
            "training": True,
            "input_video": data["video"],
            **self.get_shape(data["video"]),
            "num_samples": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "num_condition_frames": self.num_condition_frames,
        }
        positive, negative = {}, {}
        for unit in self.pipe.units:
            inputs, positive, negative = self.pipe.unit_runner(unit, self.pipe, inputs, positive, negative)
        return {**inputs, **positive}

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        return self.pipe.training_loss(**models, **inputs)

    @staticmethod
    def collate_fn(data):
        return {"video": [sample["video"] for sample in data]}
