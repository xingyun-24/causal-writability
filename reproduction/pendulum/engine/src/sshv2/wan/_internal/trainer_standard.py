import torch

from sshv2.wan._internal.sampler_standard import WanVideoPipeline
from sshv2.wan._internal.training_utils import DiffusionTrainingModule
from sshv2.wan._internal.wan_config import WanDiTConfig, WanVAEConfig


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        dit_config: WanDiTConfig | None = None,
        vae_config: WanVAEConfig | None = None,
        no_encoding: bool = False,
        num_condition_frames: int = 0,
        num_inference_steps: int | None = None,
        pipeline_type = WanVideoPipeline,
        pipeline_kwargs: dict | None = None
    ):
        super().__init__()

        self.num_inference_steps = num_inference_steps if num_inference_steps is not None \
            else getattr(dit_config, "num_inference_steps", None)

        model_configs = []
        if dit_config is not None:
            model_configs.append(dit_config.get_model_config(torch_dtype=torch.bfloat16))
        if vae_config is not None:
            model_configs.append(vae_config.get_model_config(torch_dtype=torch.bfloat16))
        else:
            default_z_dim = dit_config.in_dim if dit_config is not None else 3
            model_configs.append(WanVAEConfig.get_identity_model_config(z_dim=default_z_dim))
        self.pipe = pipeline_type.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
        )
        for k, v in (pipeline_kwargs or {}).items():
            setattr(self.pipe, k, v)
        self.pipe.pre_encoded_(no_encoding)
        if self.num_inference_steps is not None:
            self.pipe.default_num_inference_steps_(self.num_inference_steps)
        
        self.switch_pipe_to_training_mode(
            pipe=self.pipe, 
            trainable_models="dit",
            lora_base_model=None,
            lora_target_modules=None,
            lora_rank=None,
            lora_checkpoint=None,
            enable_fp8_training=False
        )
        
        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs = []
        self.max_timestep_boundary = 1.
        self.min_timestep_boundary = 0.
        self.no_encoding = no_encoding
        self.num_condition_frames = num_condition_frames

    def get_shape(self, videos):
        first = videos[0]
        if isinstance(first, torch.Tensor):
            height, width, num_frames = (
                first.shape[-2],
                first.shape[-1],
                first.shape[-3]
            )
        else:
            height, width, num_frames = (
                first[0].size[1],
                first[0].size[0],
                len(first)
            )
        if self.no_encoding:
            height = height * self.pipe.vae.upsampling_factor
            width = width * self.pipe.vae.upsampling_factor
            num_frames = self.pipe.vae.to_num_frames(num_frames)
        return {
            "height": height,
            "width": width,
            "num_frames": num_frames
        }

    def forward_preprocess(self, data):
        inputs_shared = {
            "training": True,
            "input_video": data["video"],
            **self.get_shape(data["video"]),
            "num_samples": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            "num_condition_frames": self.num_condition_frames
        }
        inputs_posi, inputs_nega = {"prompt": data.get("prompt", None)}, {}
        
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        
        return {**inputs_shared, **inputs_posi}
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss

    @staticmethod
    def collate_fn(data):
        videos, prompts = [], []
        for sample in data:
            videos.append(sample["video"])
            prompts.append(sample.get("prompt", ""))
        batch = {
            "video": videos,
            "prompt": prompts
        }
        return batch
