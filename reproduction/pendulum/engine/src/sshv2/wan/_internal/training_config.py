import os
import wandb
import yaml, json
from pathlib import Path
from typing import Literal
from accelerate import Accelerator
from pydantic import BaseModel, field_validator, model_validator
from sshv2.wan._internal.dataset import video_as_tensor_operator
from sshv2.wan._internal.wan_config import WanConfig, WanVAEConfig


class DataConfig(BaseModel):
    dataset: Path
    labels: Path | None = None
    size: tuple[int, int] = (256, 256)
    num_frames: int = 33
    load_as: Literal["frames", "tensor"] = "tensor"
    encoded: bool = False

    @model_validator(mode="after")
    def auto_labels(self):
        if self.labels is None:
            self.labels = self.dataset / "metadata.csv"
        return self
    
    def build(self):
        from diffsynth.trainers.unified_dataset import UnifiedDataset
        return UnifiedDataset(
            base_path=str(self.dataset),
            metadata_path=str(self.labels),
            data_file_keys=["image", "video"],
            main_data_operator=video_as_tensor_operator(
                base_path=str(self.dataset),
                height=self.size[0],
                width=self.size[1],
                num_frames=self.num_frames
            ),
            special_operator_map=None,
        )


class LoaderConfig(BaseModel):
    num_epochs: int = 1
    num_training_steps: int | None = None
    batch_size: int = 1
    num_workers: int = 8
    val_batch_size: int | None = None
    val_num_workers: int | None = None

    @model_validator(mode="after")
    def auto_val(self):
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        if self.val_num_workers is None:
            self.val_num_workers = self.num_workers
        return self


class OptimizerConfig(BaseModel):
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 1
    with_ema: bool = False
    ema_decay: float = 0.9999
    ema_start: int = 0
    state_file: Path | None = None


class LoggerConfig(BaseModel):
    project: str
    name: str | None = None
    ckpt_dir: Path | None = None
    out_dir: Path | None = None
    include_keys: list[str] | None = None
    exclude_keys: list[str] | None = None
    log_every: int = 10
    save_every: int | None = None
    save_last_every: int | None = None
    val_at: list[int] | None = None
    save_at: list[int] | None = None
    save_for: tuple[str, Literal["low", "high"]] | None = None
    secondary_save_for: tuple[str, Literal["low", "high"]] | None = None
    secondary_gate: tuple[str, Literal["low", "high"], float] | None = None
    rollback_on: tuple[str, Literal["low", "high"], float] | None = None
    rollback_cooldown: int = 0
    save_videos: bool = False
    val_every: int | None = None
    val_queue: bool = False
    fps: int = 15

    @field_validator("save_at")
    @classmethod
    def validate_save_at(cls, value):
        if value is None:
            return value
        if any(step <= 0 for step in value):
            raise ValueError("log.save_at entries must be positive integers")
        return sorted(set(value))

    @field_validator("val_at")
    @classmethod
    def validate_val_at(cls, value):
        if value is None:
            return value
        if any(step <= 0 for step in value):
            raise ValueError("log.val_at entries must be positive integers")
        return sorted(set(value))

    @field_validator("save_for", mode="before")
    @classmethod
    def validate_save_for(cls, value):
        if value is None:
            return value
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("log.save_for must be [metric_name, low|high]")
        metric_name, mode = value
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("log.save_for metric name must be a non-empty string")
        if mode not in ("low", "high"):
            raise ValueError("log.save_for mode must be one of: low, high")
        return (metric_name, mode)

    @field_validator("save_last_every")
    @classmethod
    def validate_save_last_every(cls, value):
        if value is None:
            return value
        if value < 0:
            raise ValueError("log.save_last_every must be >= 0")
        return value

    @field_validator("rollback_on", mode="before")
    @classmethod
    def validate_rollback_on(cls, value):
        if value is None:
            return value
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError("log.rollback_on must be [metric_name, low|high, delta]")
        metric_name, mode, delta = value
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("log.rollback_on metric name must be a non-empty string")
        if mode not in ("low", "high"):
            raise ValueError("log.rollback_on mode must be one of: low, high")
        try:
            delta = float(delta)
        except (TypeError, ValueError) as exc:
            raise ValueError("log.rollback_on delta must be numeric") from exc
        return (metric_name, mode, delta)

    @field_validator("rollback_cooldown")
    @classmethod
    def validate_rollback_cooldown(cls, value):
        if value < 0:
            raise ValueError("log.rollback_cooldown must be >= 0")
        return value


class FromFileBaseModel(BaseModel):
    @classmethod
    def from_file(cls, path: str | os.PathLike | Path):
        path = Path(path)
        ext = path.suffix
        with open(path, "r") as f:
            if ext == ".yaml":
                data = yaml.safe_load(f)
            elif ext == ".json":
                data = json.load(f)
        if data is None:
            data = {}
        return cls(**data)


class TrainingConfig(FromFileBaseModel):
    # Keep comparative pilots reproducible unless a configuration explicitly
    # requests a different initialization / data-order seed.
    seed: int = 3407
    model: WanConfig | None = None
    data: DataConfig | None = None
    val_data: DataConfig | None = None
    optimizer: OptimizerConfig = OptimizerConfig()
    loader: LoaderConfig = LoaderConfig()
    log: LoggerConfig | None = None
    

class EncodingConfig(FromFileBaseModel):
    vae: WanVAEConfig | None = None
    data: DataConfig | None = None
    loader: LoaderConfig = LoaderConfig()
    out_dir: Path | None = None
    copy_files: bool = True

    @model_validator(mode="after")
    def auto_out_dir(self):
        if self.out_dir is None and self.data is not None:
            data_dir = self.data.dataset
            num_frames, height, width = self.data.num_frames, self.data.size[0], self.data.size[1]
            self.out_dir = data_dir.with_name(f"{data_dir.name}_encoded_{num_frames}x{height}x{width}")
        return self


def setup_from(
    config_file: str | os.PathLike,
    with_wandb: bool = True,
    training_config_model: FromFileBaseModel = TrainingConfig,
    accelerator: Accelerator | None = None
) -> tuple[TrainingConfig, wandb.Run | None]:
    cfg, run = training_config_model.from_file(config_file), None
    should_init_run = with_wandb and cfg.log is not None
    if accelerator is not None and not accelerator.is_main_process:
        should_init_run = False

    if should_init_run:
        run = wandb.init(
            project=f"{cfg.log.project}",
            name=cfg.log.name,
            config=cfg.model_dump(),
            config_include_keys=cfg.log.include_keys,
            config_exclude_keys=cfg.log.exclude_keys
        )
        cfg = training_config_model(**wandb.config.as_dict())
    return cfg, run
