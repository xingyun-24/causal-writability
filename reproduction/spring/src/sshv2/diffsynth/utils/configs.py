"""Pydantic configuration models for the standalone spring experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import yaml
from typing import Any
from pydantic import BaseModel, field_validator, model_validator

from sshv2.diffsynth.configs.wan_config import WanConfig


class DataConfig(BaseModel):
    dataset: Path
    labels: Path | None = None
    size: tuple[int, int] = (128, 128)
    num_frames: int
    load_as: Literal["frames", "tensor"] = "tensor"
    encoded: bool = True

    @model_validator(mode="after")
    def auto_labels(self):
        if self.labels is None:
            self.labels = self.dataset / "metadata.csv"
        return self

    def build(self):
        from diffsynth.trainers.unified_dataset import UnifiedDataset
        from sshv2.diffsynth.trainers.dataset import video_as_tensor_operator

        return UnifiedDataset(
            base_path=str(self.dataset),
            metadata_path=str(self.labels),
            data_file_keys=["image", "video"],
            main_data_operator=video_as_tensor_operator(
                base_path=str(self.dataset),
                height=self.size[0],
                width=self.size[1],
                num_frames=self.num_frames,
            ),
        )


class LoaderConfig(BaseModel):
    num_epochs: int = 1
    num_training_steps: int | None = None
    batch_size: int = 1
    num_workers: int = 4
    val_batch_size: int | None = None
    val_num_workers: int | None = None

    @model_validator(mode="after")
    def auto_validation_loader(self):
        if self.val_batch_size is None:
            self.val_batch_size = self.batch_size
        if self.val_num_workers is None:
            self.val_num_workers = self.num_workers
        return self


class OptimizerConfig(BaseModel):
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    gradient_accumulation_steps: int = 1
    with_ema: bool = False
    ema_decay: float = 0.9999
    ema_start: int = 0
    state_file: Path | None = None


class LoggerConfig(BaseModel):
    project: str
    name: str | None = None
    ckpt_dir: Path
    out_dir: Path
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

    @field_validator("save_at", "val_at")
    @classmethod
    def validate_step_lists(cls, value):
        if value is None:
            return value
        if any(step <= 0 for step in value):
            raise ValueError("checkpoint/validation step entries must be positive integers")
        return sorted(set(value))

    @field_validator("save_last_every")
    @classmethod
    def validate_save_last_every(cls, value):
        if value is not None and value < 0:
            raise ValueError("log.save_last_every must be >= 0")
        return value


class FromFileBaseModel(BaseModel):
    @classmethod
    def from_file(cls, path: str | os.PathLike | Path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix == ".yaml":
                data = yaml.safe_load(handle)
            elif path.suffix == ".json":
                data = json.load(handle)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls(**(data or {}))


class TrainingConfig(FromFileBaseModel):
    seed: int = 3407
    model: WanConfig
    data: DataConfig
    optimizer: OptimizerConfig = OptimizerConfig()
    loader: LoaderConfig = LoaderConfig()
    log: LoggerConfig


def setup_from(
    config_file: str | os.PathLike,
    with_wandb: bool = True,
    training_config_model: type[FromFileBaseModel] = TrainingConfig,
    accelerator: Any | None = None,
):
    cfg = training_config_model.from_file(config_file)
    run = None
    should_init = with_wandb and (accelerator is None or accelerator.is_main_process)
    if should_init:
        import wandb

        run = wandb.init(
            project=cfg.log.project,
            name=cfg.log.name,
            config=cfg.model_dump(),
            config_include_keys=cfg.log.include_keys,
            config_exclude_keys=cfg.log.exclude_keys,
        )
        cfg = training_config_model(**wandb.config.as_dict())
    return cfg, run
