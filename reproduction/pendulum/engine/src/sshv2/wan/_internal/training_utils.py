import os
import math
import time
import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from collections import deque
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch.optim.swa_utils import AveragedModel
from accelerate.utils import DistributedDataParallelKwargs
from safetensors.torch import load_file as load_safetensors

from diffsynth.trainers.utils import *
from sshv2.wan._internal.training_config import TrainingConfig


def infer_batch_size(data, fallback=1):
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0] if value.ndim > 0 else fallback
            if isinstance(value, (list, tuple)):
                return len(value)
    elif isinstance(data, torch.Tensor):
        return data.shape[0] if data.ndim > 0 else fallback
    elif isinstance(data, (list, tuple)):
        return len(data)
    return fallback


class TrainLogger:
    def __init__(self, model_logger: ModelLogger, save_every=None, save_at=None, save_last_every=None):
        self.model_logger = model_logger
        self.save_every = save_every
        self.save_last_every = save_last_every
        self.save_at = set(save_at or [])
        self.num_steps = 0
        self.last_validation_step = None
        self.last_saved_step = None
        self.last_refreshed_step = None

    def _save_model(self, accelerator, model, step, avg_model=None, state=None):
        self.model_logger.save_model(accelerator, model, f"step-{step}.safetensors")
        if avg_model is not None:
            avg_model = avg_model.module if isinstance(avg_model, AveragedModel) else avg_model
            self.model_logger.save_model(accelerator, avg_model, f"step-{step}-avg.safetensors")
        if state is not None:
            self.save_state(accelerator, state, f"step-{step}-state.pt")
        self.last_saved_step = step

    def on_step_end(self, accelerator, model, avg_model=None, state=None):
        self.num_steps += 1
        should_save = False
        if self.save_every is not None and self.save_every > 0 and self.num_steps % self.save_every == 0:
            should_save = True
        if self.save_at and self.num_steps in self.save_at:
            should_save = True
        if should_save:
            self._save_model(accelerator, model, self.num_steps, avg_model=avg_model, state=state)
        if self.save_last_every is not None and self.save_last_every > 0 and self.num_steps % self.save_last_every == 0:
            self.save_last(accelerator, model, avg_model=avg_model, state=state, step=self.num_steps)

    def save_state(self, accelerator, state, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(self.model_logger.output_path, exist_ok=True)
            path = os.path.join(self.model_logger.output_path, file_name)
            accelerator.save(state, path, safe_serialization=False)

    def save_last(self, accelerator, model, avg_model=None, state=None, step=None):
        step = self.num_steps if step is None else step
        if step <= 0 or self.last_refreshed_step == step:
            return
        self.model_logger.save_model(accelerator, model, "last.safetensors")
        if state is not None:
            self.save_state(accelerator, state, "last-state.pt")
        if avg_model is not None:
            avg_model = avg_model.module if isinstance(avg_model, AveragedModel) else avg_model
            self.model_logger.save_model(accelerator, avg_model, "last-avg.safetensors")
        self.last_refreshed_step = step

    def on_epoch_end(self, accelerator, model, epoch_id, avg_model=None, state=None):
        self.model_logger.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")
        if avg_model is not None:
            avg_model = avg_model.module if isinstance(avg_model, AveragedModel) else avg_model
            self.model_logger.save_model(accelerator, avg_model, f"epoch-{epoch_id}-avg.safetensors")
        if state is not None:
            self.save_state(accelerator, state, f"epoch-{epoch_id}-state.pt")

    def on_training_end(self, accelerator, model, avg_model=None, state=None):
        if self.num_steps <= 0:
            return
        if self.last_saved_step != self.num_steps:
            self._save_model(accelerator, model, self.num_steps, avg_model=avg_model, state=state)
        self.save_last(accelerator, model, avg_model=avg_model, state=state, step=self.num_steps)


class Validator:
    def __init__(
        self,
        on_validation: Callable | None = None,
        wandb_run: wandb.Run | None = None,
        val_every: int | None = None,
        val_at: list[int] | None = None,
        train_logger: TrainLogger | None = None,
        save_for: tuple[str, str] | None = None,
        secondary_save_for: tuple[str, str] | None = None,
        secondary_gate: tuple[str, str, float] | None = None,
        rollback_on: tuple[str, str, float] | None = None,
        rollback_cooldown: int = 0,
    ):
        self.on_validation = on_validation
        self.wandb_run = wandb_run
        self.val_every = val_every
        self.val_at = set(val_at or [])
        self.train_logger = train_logger
        self.save_for = tuple(save_for) if save_for is not None else None
        self.secondary_save_for = tuple(secondary_save_for) if secondary_save_for is not None else None
        self.secondary_gate = tuple(secondary_gate) if secondary_gate is not None else None
        self.secondary_top_metric_value = None
        self.rollback_on = tuple(rollback_on) if rollback_on is not None else None
        self.rollback_cooldown = int(rollback_cooldown)
        self.num_steps = 0
        self.top_metric_value = None
        self.top_metric_step = None
        self.rollback_reference_value = None
        self.last_rollback_step = None
        self.last_validation_step = None
        self.did_warn_missing_metric = False
        self.did_warn_missing_rollback_metric = False
        self.did_warn_missing_rollback_reference = False
        self.did_warn_missing_top_checkpoint = False

    def _try_log(self, metrics, step: int, prefix: str = ""):
        if metrics is None or self.wandb_run is None:
            return
        self.wandb_run.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def on_step_end(self, accelerator=None, model=None, avg_model=None, state=None):
        self.num_steps += 1
        if self.on_validation is None or accelerator is None or model is None:
            return False
        should_validate = False
        if self.val_every is not None and self.val_every > 0 and self.num_steps % self.val_every == 0:
            should_validate = True
        if self.val_at and self.num_steps in self.val_at:
            should_validate = True
        if not should_validate:
            return False
        return self._run_validation(accelerator, model, avg_model, state=state)

    def on_epoch_end(self, accelerator=None, model=None, avg_model=None, state=None):
        if self.on_validation is None or accelerator is None or model is None:
            return False
        # `val_at`/`val_every` are step schedules.  A tiny overfit dataset can
        # be exactly one batch, so treating every epoch as an extra validation
        # would turn a 1,000-step run into 1,000 expensive sampling passes.
        if self.val_every is not None or self.val_at:
            return False
        return self._run_validation(accelerator, model, avg_model, state=state)

    def on_training_end(self, accelerator=None, model=None, avg_model=None, state=None):
        if self.on_validation is None or accelerator is None or model is None:
            return False
        if self.last_validation_step == self.num_steps:
            return False
        return self._run_validation(accelerator, model, avg_model, state=state)

    def _build_metric_aliases(self, metrics: dict | None, *, avg_prefix: bool = False):
        aliases = {}
        if metrics is None:
            return aliases
        for key, value in metrics.items():
            if not isinstance(key, str):
                continue
            if avg_prefix:
                aliases[f"avg_{key}"] = value
                if not key.startswith("val/"):
                    aliases[f"avg_val/{key}"] = value
            else:
                aliases[key] = value
                if not key.startswith("val/"):
                    aliases[f"val/{key}"] = value
        return aliases

    def _select_monitored_metric(self, base_metrics: dict | None, avg_metrics: dict | None):
        if self.save_for is None:
            return None, None
        metric_name, _ = self.save_for
        return self._select_metric_by_name(metric_name, base_metrics, avg_metrics)

    def _select_metric_by_name(self, metric_name: str, base_metrics: dict | None, avg_metrics: dict | None):
        avg_aliases = self._build_metric_aliases(avg_metrics, avg_prefix=True)
        if metric_name in avg_aliases:
            return "avg", avg_aliases[metric_name]
        base_aliases = self._build_metric_aliases(base_metrics, avg_prefix=False)
        if metric_name in base_aliases:
            return "base", base_aliases[metric_name]
        return None, None

    def _as_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.item()
        try:
            return float(value)
        except Exception as _:
            return None

    def _did_improve(self, value: float):
        if self.save_for is None:
            return False
        _, mode = self.save_for
        if self.top_metric_value is None:
            return True
        if mode == "low":
            return value < self.top_metric_value
        return value > self.top_metric_value

    @staticmethod
    def _did_improve_secondary(value: float, best: float | None, mode: str):
        return best is None or (value < best if mode == "low" else value > best)

    @staticmethod
    def _passes_gate(value: float | None, mode: str, threshold: float):
        return value is not None and (value >= threshold if mode == "high" else value <= threshold)

    def _should_rollback(self, value: float, reference_value: float):
        if self.rollback_on is None:
            return False
        _, mode, delta = self.rollback_on
        if mode == "low":
            return value > reference_value + delta
        return value < reference_value - delta

    def _run_validation(self, accelerator, model, avg_model=None, state=None):
        self.last_validation_step = self.num_steps
        accelerator.wait_for_everyone()
        base_model = accelerator.unwrap_model(model)
        base_metrics = self.on_validation(
            accelerator=accelerator, pipe=base_model.pipe,
            model_tag="base", step=self.num_steps
        )
        if accelerator.is_main_process:
            self._try_log(base_metrics, self.num_steps)

        avg_metrics = None
        if avg_model is not None:
            avg_model = avg_model.module if isinstance(avg_model, AveragedModel) else avg_model
            inference_dtype = accelerator.unwrap_model(model).pipe.torch_dtype
            avg_model.pipe.to(dtype=inference_dtype)
            try:
                avg_metrics = self.on_validation(
                    accelerator=accelerator, pipe=avg_model.pipe,
                    model_tag="avg", step=self.num_steps
                )
            finally:
                avg_model.pipe.to(dtype=torch.float32)
            if accelerator.is_main_process:
                self._try_log(avg_metrics, self.num_steps, prefix="avg_")
        
        if self.train_logger is not None:
            self.train_logger.save_last(accelerator, model, avg_model=avg_model, state=state, step=self.num_steps)

        save_mode = 0
        secondary_save_mode = 0
        rollback_mode = 0
        rollback_metric = None
        if self.save_for is not None and self.train_logger is not None and accelerator.is_main_process:
            model_type, metric_value = self._select_monitored_metric(base_metrics, avg_metrics)
            metric_value = self._as_scalar(metric_value)
            if model_type is None and not self.did_warn_missing_metric:
                metric_name, _ = self.save_for
                print(f"Warning: log.save_for metric '{metric_name}' was not found in validation metrics.")
                self.did_warn_missing_metric = True
            if metric_value is not None and self._did_improve(metric_value):
                self.top_metric_value = metric_value
                self.top_metric_step = self.num_steps
                save_mode = 2 if model_type == "avg" else 1

        if self.secondary_save_for is not None and self.train_logger is not None and accelerator.is_main_process:
            metric_name, metric_mode = self.secondary_save_for
            secondary_type, secondary_value = self._select_metric_by_name(metric_name, base_metrics, avg_metrics)
            secondary_value = self._as_scalar(secondary_value)
            passes_gate = True
            if self.secondary_gate is not None:
                gate_name, gate_mode, gate_threshold = self.secondary_gate
                _, gate_value = self._select_metric_by_name(gate_name, base_metrics, avg_metrics)
                passes_gate = self._passes_gate(self._as_scalar(gate_value), gate_mode, gate_threshold)
            if passes_gate and secondary_value is not None and self._did_improve_secondary(
                secondary_value, self.secondary_top_metric_value, metric_mode
            ):
                self.secondary_top_metric_value = secondary_value
                secondary_save_mode = 2 if secondary_type == "avg" else 1

        if self.rollback_on is not None and accelerator.is_main_process:
            rollback_metric_name, _, _ = self.rollback_on
            model_type, rollback_metric = self._select_metric_by_name(rollback_metric_name, base_metrics, avg_metrics)
            rollback_metric = self._as_scalar(rollback_metric)
            if model_type is None and not self.did_warn_missing_rollback_metric:
                print(f"Warning: log.rollback_on metric '{rollback_metric_name}' was not found in validation metrics.")
                self.did_warn_missing_rollback_metric = True
            elif save_mode in (1, 2) and rollback_metric is not None:
                self.rollback_reference_value = rollback_metric
                self.did_warn_missing_rollback_reference = False
            elif rollback_metric is not None:
                cooldown_ready = (
                    self.last_rollback_step is None
                    or (self.num_steps - self.last_rollback_step) > self.rollback_cooldown
                )
                if self.top_metric_step is None:
                    if not self.did_warn_missing_top_checkpoint:
                        print("Warning: rollback requested but no top checkpoint has been saved yet.")
                        self.did_warn_missing_top_checkpoint = True
                elif self.rollback_reference_value is None:
                    if not self.did_warn_missing_rollback_reference:
                        print("Warning: rollback requested but no rollback reference metric has been saved yet.")
                        self.did_warn_missing_rollback_reference = True
                elif cooldown_ready and self.top_metric_step < self.num_steps and self._should_rollback(rollback_metric, self.rollback_reference_value):
                    rollback_mode = 1

        action_tensor = torch.tensor([save_mode, secondary_save_mode, rollback_mode], device=accelerator.device, dtype=torch.int32)
        if accelerator.num_processes > 1:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(action_tensor, src=0)
            else:
                action_tensor.zero_()
        save_mode, secondary_save_mode, rollback_mode = [int(v) for v in action_tensor.tolist()]

        if self.train_logger is not None and save_mode in (1, 2):
            self.train_logger.model_logger.save_model(accelerator, model, "top.safetensors")
            if state is not None:
                self.train_logger.save_state(accelerator, state, "top-state.pt")
        if self.train_logger is not None and save_mode == 2 and avg_model is not None:
            self.train_logger.model_logger.save_model(accelerator, avg_model, "top-avg.safetensors")
            if state is not None:
                self.train_logger.save_state(accelerator, state, "top-avg-state.pt")
        if self.train_logger is not None and secondary_save_mode in (1, 2):
            secondary_model = avg_model if secondary_save_mode == 2 and avg_model is not None else model
            self.train_logger.model_logger.save_model(accelerator, secondary_model, "swap.safetensors")
            if state is not None:
                self.train_logger.save_state(accelerator, state, "swap-state.pt")
        accelerator.wait_for_everyone()
        return rollback_mode == 1


def launch_training_task(
    dataset: Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    on_validation: Callable | None = None,
    accelerator: Accelerator | None = None,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    num_epochs: int = 1,
    num_training_steps: int | None = None,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    num_workers: int = 8,
    with_ema: bool = False,
    ema_decay: float = 0.9999,
    ema_start: int = 0,
    save_every: int | None = None,
    save_last_every: int | None = None,
    val_at: list[int] | None = None,
    save_at: list[int] | None = None,
    save_for: tuple[str, str] | None = None,
    secondary_save_for: tuple[str, str] | None = None,
    secondary_gate: tuple[str, str, float] | None = None,
    rollback_on: tuple[str, str, float] | None = None,
    rollback_cooldown: int = 0,
    val_every: int | None = None,
    log_every: int = 10,
    ema_ckpt_file: str | os.PathLike | Path = None,
    state_ckpt_file: str | os.PathLike | Path = None,
    find_unused_parameters: bool = False,
    wandb_run: wandb.Run | None = None,
    config: TrainingConfig | None = None,
):
    if config is not None:
        learning_rate = config.optimizer.learning_rate
        weight_decay = config.optimizer.weight_decay
        num_epochs = config.loader.num_epochs
        num_training_steps = config.loader.num_training_steps
        batch_size = config.loader.batch_size
        gradient_accumulation_steps = config.optimizer.gradient_accumulation_steps
        num_workers = config.loader.num_workers
        with_ema = config.optimizer.with_ema
        ema_decay = config.optimizer.ema_decay
        ema_start = config.optimizer.ema_start
        save_every = config.log.save_every
        save_last_every = config.log.save_last_every
        val_at = config.log.val_at
        save_at = config.log.save_at
        save_for = config.log.save_for
        secondary_save_for = config.log.secondary_save_for
        secondary_gate = config.log.secondary_gate
        rollback_on = config.log.rollback_on
        rollback_cooldown = config.log.rollback_cooldown
        val_every = config.log.val_every
        log_every = config.log.log_every

    save_at = set(save_at or [])
    _save_every = save_every if (save_every is not None and save_every > 0) else None
    _save_last_every = save_last_every if (save_last_every is not None and save_last_every > 0) else None
    train_logger = TrainLogger(
        model_logger,
        save_every=_save_every,
        save_at=save_at,
        save_last_every=_save_last_every,
    )
    validator = Validator(
        on_validation=on_validation, 
        wandb_run=wandb_run, val_every=val_every, val_at=val_at,
        train_logger=train_logger, save_for=save_for,
        secondary_save_for=secondary_save_for, secondary_gate=secondary_gate,
        rollback_on=rollback_on, rollback_cooldown=rollback_cooldown,
    )

    ema_ckpt = load_safetensors(ema_ckpt_file) if ema_ckpt_file is not None else None
    state_ckpt = torch.load(state_ckpt_file, map_location="cpu") if state_ckpt_file is not None else None

    if accelerator is None:
        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        )

    num_processes = max(int(getattr(accelerator, "num_processes", 1)), 1)
    if num_processes > 1:
        if batch_size % num_processes != 0:
            raise ValueError(f"`loader.batch_size` ({batch_size}) must be divisible by the number of GPUs ({num_processes}) with multi-GPU training")
        batch_size = batch_size // num_processes

    collate_fn = model.collate_fn if hasattr(model, "collate_fn") else lambda x: x[0]
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if state_ckpt is not None:
        if "optimizer" in state_ckpt:
            print(f"Loading optimizer from: {state_ckpt_file}")
            optimizer.load_state_dict(state_ckpt["optimizer"])
        if "scheduler" in state_ckpt:
            print(f"Loading scheduler from: {state_ckpt_file}")
            scheduler.load_state_dict(state_ckpt["scheduler"])

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    num_epoch_steps = math.ceil(len(dataloader) / gradient_accumulation_steps)
    num_training_steps = num_training_steps or num_epochs * num_epoch_steps

    last_step = int(state_ckpt.get("num_training_steps", 0)) if state_ckpt is not None else 0
    i = min(last_step, num_training_steps)
    epoch_id = i // num_epoch_steps if num_epoch_steps > 0 else 0

    train_logger.num_steps = i
    validator.num_steps = i

    ema = None

    def init_ema():
        avg_fn = lambda prev, next, _: ema_decay * prev + (1 - ema_decay) * next
        ema = AveragedModel(
            accelerator.unwrap_model(model),
            avg_fn=avg_fn,
            use_buffers=True
        ).to(accelerator.device)
        ema.module.to(dtype=torch.float32)
        if ema_ckpt_file is not None:
            ema.module.pipe.dit.load_state_dict(ema_ckpt)
        return ema

    def state():
        return {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "num_training_steps": i,
            "epoch_id": epoch_id
        }

    def _load_dit_checkpoint(file_name: str, target_model):
        path = Path(model_logger.output_path) / file_name
        if not path.exists():
            return False
        state_dict = load_safetensors(path)
        missing, unexpected = target_model.pipe.dit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing {len(missing)} keys while loading {path}")
        if unexpected:
            print(f"Warning: unexpected {len(unexpected)} keys while loading {path}")
        return True

    def restore_top_checkpoint():
        nonlocal i, epoch_id
        top_model_path = Path(model_logger.output_path) / "top.safetensors"
        top_state_path = Path(model_logger.output_path) / "top-state.pt"
        if not top_model_path.exists() or not top_state_path.exists():
            return False

        base_model = accelerator.unwrap_model(model)
        if not _load_dit_checkpoint("top.safetensors", base_model):
            return False

        top_state = torch.load(top_state_path, map_location="cpu")
        if "optimizer" in top_state:
            optimizer.load_state_dict(top_state["optimizer"])
        if "scheduler" in top_state:
            scheduler.load_state_dict(top_state["scheduler"])

        i = int(top_state.get("num_training_steps", i))
        epoch_id = int(top_state.get("epoch_id", epoch_id))
        train_logger.num_steps = i
        train_logger.last_saved_step = None
        train_logger.last_refreshed_step = None
        validator.num_steps = i
        validator.last_rollback_step = i

        if ema is not None:
            if not _load_dit_checkpoint("top-avg.safetensors", ema.module):
                _load_dit_checkpoint("top.safetensors", ema.module)
            ema.module.to(dtype=torch.float32)
        train_logger.save_last(accelerator, model, avg_model=ema, state=state(), step=i)
        return True

    def restart_wandb_run():
        nonlocal wandb_run
        if not accelerator.is_main_process or wandb_run is None or config is None or config.log is None:
            return
        try:
            wandb_run.finish()
        except Exception as exc:
            print(f"Failed to finish W&B run before restart: {exc}")
        wandb_run = wandb.init(
            project=f"{config.log.project}",
            name=config.log.name,
            config=config.model_dump(),
            config_include_keys=config.log.include_keys,
            config_exclude_keys=config.log.exclude_keys,
        )
        validator.wandb_run = wandb_run

    log_every = max(int(log_every), 1)
    samples = total_samples = 0
    throughput_window = deque(maxlen=32)
    it_s, samples_s = 0.0, 0.0

    pbar = tqdm(total=num_training_steps, desc="Training", initial=i, disable=(not accelerator.is_main_process))
    while i < num_training_steps:
        for data in dataloader:
            should_restart = False
            samples += infer_batch_size(data, fallback=batch_size)
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if getattr(dataset, "load_from_cache", False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()

                    i += 1
                    if num_epoch_steps > 0:
                        epoch_id = (i - 1) // num_epoch_steps
                    pbar.update(1)

                    total_samples += samples * accelerator.num_processes
                    samples = 0
                    now = time.perf_counter()
                    throughput_window.append((now, i, total_samples))
                    if len(throughput_window) >= 2:
                        t0, i0, s0 = throughput_window[0]
                        dt = now - t0
                        if dt > 0:
                            it_s = (i - i0) / dt
                            samples_s = (total_samples - s0) / dt
                            pbar.set_postfix({
                                "it/s": f"{it_s:.2f}",
                                "samples/s": f"{samples_s:.2f}",
                            }, refresh=False)

                    if with_ema and i >= ema_start:
                        if ema is None:
                            ema = init_ema()
                        ema.update_parameters(accelerator.unwrap_model(model))

                    should_log = i % log_every == 0
                    if should_log and (wandb_run is not None or accelerator.num_processes > 1):
                        global_loss = accelerator.gather_for_metrics(loss.detach()).mean()
                        if accelerator.is_main_process and wandb_run is not None:
                            wandb_run.log({
                                "train/loss": global_loss.item(),
                                "train/lr": optimizer.param_groups[0]["lr"],
                                "epoch": epoch_id
                            }, step=i)
                    _state = state()
                    train_logger.on_step_end(accelerator, model, avg_model=ema, state=_state)
                    should_restart = validator.on_step_end(accelerator, model, avg_model=ema, state=_state)

                # A dataset smaller than a training batch has one epoch per
                # optimizer step.  When `save_last_every` is configured it is
                # the explicit checkpoint schedule; do not additionally emit
                # full model + EMA + optimizer snapshots at every epoch.
                if save_every is None and not save_at and save_last_every is None and i % num_epoch_steps == 0:
                    train_logger.on_epoch_end(accelerator, model, avg_model=ema, state=state(), epoch_id=epoch_id)

                if val_every is None and i % num_epoch_steps == 0:
                    should_restart = validator.on_epoch_end(accelerator, model, avg_model=ema, state=state()) or should_restart

                if should_restart:
                    if restore_top_checkpoint():
                        restart_wandb_run()
                        throughput_window.clear()
                        samples = 0
                        if accelerator.is_main_process:
                            pbar.n = i
                            pbar.last_print_n = i
                            pbar.refresh()
                            print(f"Rolled back training to top checkpoint at step {i}.")
                    else:
                        print("Warning: rollback requested but top checkpoint files could not be loaded.")
                    break

                if i >= num_training_steps:
                    break

    validator.on_training_end(accelerator, model, avg_model=ema, state=state())
    train_logger.on_training_end(accelerator, model, avg_model=ema, state=state())

    pbar.close()
