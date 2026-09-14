"""Direct training entry point for the existing physics-shortcut experiments."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch._dynamo
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from sshv2.wan._internal.compat_training_config import (
    CompatibilityTrainingConfig,
)
from sshv2.wan._internal.trainer_compat import (
    CompatibilityWanTrainingModule,
)
from sshv2.wan._internal.training_utils import ModelLogger, launch_training_task
from sshv2.wan._internal.training_config import setup_from


torch._dynamo.config.optimize_ddp = False


def _resolve_resume_dir(
    value: Path | bool,
    cfg: CompatibilityTrainingConfig,
    run_name: str,
    flag_name: str,
) -> Path:
    if value is True:
        if cfg.log is None or cfg.log.ckpt_dir is None:
            raise ValueError(
                f"Empty `{flag_name}`; log.ckpt_dir is missing"
            )
        return cfg.log.ckpt_dir / run_name
    return Path(value)


def _apply_resume_checkpoint(
    cfg: CompatibilityTrainingConfig,
    resume_dir: Path,
    prefix: str,
) -> None:
    ckpt_path = resume_dir / f"{prefix}.safetensors"
    state_path = resume_dir / f"{prefix}-state.pt"
    avg_ckpt_path = resume_dir / f"{prefix}-avg.safetensors"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing optimizer state file: {state_path}")
    if cfg.model is None:
        raise ValueError("The training config does not define model")

    cfg.model.dit.ckpt_file = ckpt_path
    cfg.optimizer.state_file = state_path
    cfg.model.dit.avg_ckpt_file = (
        avg_ckpt_path if avg_ckpt_path.exists() else None
    )


def train(
    config_file: Path,
    *,
    resume: Path | bool | None = None,
    resume_top: Path | bool | None = None,
    steps: int | None = None,
    val_every: int | None = None,
    ckpt_dir: Path | None = None,
    out_dir: Path | None = None,
) -> None:
    """Train directly from one resolved experiment configuration."""
    pre_cfg, _ = setup_from(
        config_file,
        with_wandb=False,
        training_config_model=CompatibilityTrainingConfig,
    )
    multi_gpu = (
        torch.cuda.is_available()
        and int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=(
            pre_cfg.optimizer.gradient_accumulation_steps
        ),
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=multi_gpu,
            )
        ],
    )
    cfg, run = setup_from(
        config_file,
        training_config_model=CompatibilityTrainingConfig,
        accelerator=accelerator,
    )
    set_seed(cfg.seed, device_specific=multi_gpu)

    if cfg.model is None or cfg.data is None or cfg.log is None:
        raise ValueError(
            "The config must define model, data, and log sections"
        )
    if cfg.val_data is not None:
        raise ValueError(
            "Experiment validation is explicit; run its evaluation.py "
            "after training instead of configuring val_data"
        )
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        cfg.loader.num_training_steps = steps
    if val_every is not None:
        if val_every <= 0:
            raise ValueError("val_every must be positive")
        cfg.log.val_every = val_every
        cfg.log.val_at = None
    if ckpt_dir is not None:
        cfg.log.ckpt_dir = ckpt_dir
    if out_dir is not None:
        cfg.log.out_dir = out_dir

    configured_name = cfg.log.name
    run_name = (
        run.name
        if run is not None
        else (configured_name or "offline")
    )
    if resume is not None:
        _apply_resume_checkpoint(
            cfg,
            _resolve_resume_dir(
                resume,
                cfg,
                run_name,
                "-r/--resume",
            ),
            "last",
        )
    elif resume_top is not None:
        _apply_resume_checkpoint(
            cfg,
            _resolve_resume_dir(
                resume_top,
                cfg,
                run_name,
                "--resume-top",
            ),
            "top",
        )

    dataset = cfg.data.build()
    model = CompatibilityWanTrainingModule(
        dit_config=cfg.model.dit,
        vae_config=cfg.model.vae,
        no_encoding=cfg.data.encoded,
        num_condition_frames=cfg.model.num_condition_frames,
        num_inference_steps=cfg.model.dit.num_inference_steps,
        pipeline_type=cfg.model.pipe,
        pipeline_kwargs=cfg.model.pipe_kwargs,
    )
    model_logger = ModelLogger(
        cfg.log.ckpt_dir / run_name,
        remove_prefix_in_ckpt="pipe.dit.",
    )
    launch_training_task(
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        accelerator=accelerator,
        on_validation=None,
        wandb_run=run,
        config=cfg,
        ema_ckpt_file=cfg.model.dit.avg_ckpt_file,
        state_ckpt_file=cfg.optimizer.state_file,
        find_unused_parameters=multi_gpu,
    )
