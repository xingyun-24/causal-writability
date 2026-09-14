"""Direct training entry point for the standard Wan profile."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch._dynamo
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from sshv2.wan._internal.standard_training_config import (
    StandardTrainingConfig,
)
from sshv2.wan._internal.training_utils import ModelLogger, launch_training_task
from sshv2.wan._internal.trainer_standard import WanTrainingModule
from sshv2.wan._internal.training_config import setup_from


torch._dynamo.config.optimize_ddp = False


def _resolve_resume_dir(
    value: Path | bool,
    cfg: StandardTrainingConfig,
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
    cfg: StandardTrainingConfig,
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
    ckpt_dir: Path | None = None,
    out_dir: Path | None = None,
    seed: int | None = None,
    with_wandb: bool = True,
) -> None:
    pre_cfg, _ = setup_from(
        config_file,
        with_wandb=False,
        training_config_model=StandardTrainingConfig,
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
        with_wandb=with_wandb,
        training_config_model=StandardTrainingConfig,
        accelerator=accelerator,
    )
    if seed is not None:
        cfg.seed = seed
    set_seed(cfg.seed, device_specific=multi_gpu)

    if not cfg.data.encoded:
        raise ValueError(
            "Spring training expects data.encoded: true"
        )
    if steps is not None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        cfg.loader.num_training_steps = steps
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
    model = WanTrainingModule(
        dit_config=cfg.model.dit,
        vae_config=cfg.model.vae,
        no_encoding=True,
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
