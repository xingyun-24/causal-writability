"""Train one short-history or long-history spring model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch._dynamo
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed

from sshv2.diffsynth.trainers.utils import ModelLogger, launch_training_task
from sshv2.diffsynth.trainers.wan_video_train import WanTrainingModule
from sshv2.diffsynth.utils.configs import setup_from
from sshv2.utils.spring_configs import SpringTrainingConfig


torch._dynamo.config.optimize_ddp = False


def resolve_resume_dir(arg_value: Path | bool, cfg: SpringTrainingConfig, run_name: str, flag_name: str) -> Path:
    if arg_value is True:
        if cfg.log is None or cfg.log.ckpt_dir is None:
            raise ValueError(f"Empty `{flag_name}`; `log.ckpt_dir` is missing from the config.")
        return cfg.log.ckpt_dir / run_name
    return Path(arg_value)


def apply_resume_checkpoint(cfg: SpringTrainingConfig, resume_dir: Path, prefix: str) -> None:
    ckpt_path = resume_dir / f"{prefix}.safetensors"
    state_path = resume_dir / f"{prefix}-state.pt"
    avg_ckpt_path = resume_dir / f"{prefix}-avg.safetensors"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing optimizer state file: {state_path}")

    if cfg.model is None:
        raise ValueError("The training config does not define `model`.")
    cfg.model.dit.ckpt_file = ckpt_path
    cfg.optimizer.state_file = state_path
    cfg.model.dit.avg_ckpt_file = avg_ckpt_path if avg_ckpt_path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("-r", "--resume", nargs="?", type=Path, const=True, default=None)
    resume_group.add_argument("--resume-top", nargs="?", type=Path, const=True, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    pre_cfg, _ = setup_from(
        args.config,
        with_wandb=False,
        training_config_model=SpringTrainingConfig,
    )

    multi_gpu = torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1
    accelerator = Accelerator(
        gradient_accumulation_steps=pre_cfg.optimizer.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=multi_gpu)],
    )

    cfg, run = setup_from(
        args.config,
        with_wandb=not args.no_wandb,
        training_config_model=SpringTrainingConfig,
        accelerator=accelerator,
    )
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed, device_specific=multi_gpu)

    if cfg.model is None or cfg.data is None or cfg.log is None:
        raise ValueError("The config must define model, data, and log sections.")
    if not cfg.data.encoded:
        raise ValueError("Spring training expects pre-encoded latent tensors (`data.encoded: true`).")

    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        cfg.loader.num_training_steps = args.steps
    if args.ckpt_dir is not None:
        cfg.log.ckpt_dir = args.ckpt_dir
    if args.out_dir is not None:
        cfg.log.out_dir = args.out_dir

    config_run_name = cfg.log.name
    run_name = run.name if run is not None else (config_run_name or "offline")

    if args.resume is not None:
        apply_resume_checkpoint(cfg, resolve_resume_dir(args.resume, cfg, run_name, "-r/--resume"), "last")
    elif args.resume_top is not None:
        apply_resume_checkpoint(cfg, resolve_resume_dir(args.resume_top, cfg, run_name, "--resume-top"), "top")

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

    model_logger = ModelLogger(cfg.log.ckpt_dir / run_name, remove_prefix_in_ckpt="pipe.dit.")
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


if __name__ == "__main__":
    main()
