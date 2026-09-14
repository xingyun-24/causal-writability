"""Condition-token residual patching utilities for the Spring V4 Long DiT.

The patch coordinate has three axes:

* Flow-matching sampling step: all configured steps are patched.
* DiT depth: one post-embedding or post-block residual location per run.
* Video token position: only the temporal prefix corresponding to condition
  latent frames is replaced; receiver target tokens remain untouched.

The module deliberately does not modify the production Wan pipeline.  It
implements a latent-only sampler matching that pipeline and exposes a narrow
activation-bank interface for record/inject runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
from einops import rearrange

from diffsynth.pipelines.wan_video_new import sinusoidal_embedding_1d
from sshv2.interpretability.p1_position_probe import (
    FEATURE_LAYOUT,
    FUTURE_LATENT_START,
    FUTURE_LATENT_STOP,
    TARGET_DESCRIPTION,
    normalize_and_flatten,
)
from sshv2.simulation.spring_shortcuts_v1 import classify_omega, fit_free_shm_frequency

LayerKey = int  # -1 means post-patch-embedding; 0..N-1 mean post-block output.
PatchKind = Literal["condition", "full"]


@dataclass
class ActivationBank:
    """CPU activation cache indexed by patch kind, layer, and sampling step."""

    condition: dict[LayerKey, list[torch.Tensor]] = field(default_factory=dict)
    full: dict[LayerKey, list[torch.Tensor]] = field(default_factory=dict)

    def record(self, kind: PatchKind, layer: LayerKey, value: torch.Tensor) -> None:
        target = self.condition if kind == "condition" else self.full
        target.setdefault(layer, []).append(value.detach().to(device="cpu", copy=True))

    def get(self, kind: PatchKind, layer: LayerKey, step: int) -> torch.Tensor:
        source = self.condition if kind == "condition" else self.full
        if layer not in source:
            raise KeyError(f"No {kind} cache for layer {layer}")
        values = source[layer]
        if not 0 <= step < len(values):
            raise IndexError(
                f"No {kind} cache for layer {layer}, step {step}; "
                f"cached steps={len(values)}"
            )
        return values[step]

    def assert_steps(self, layers: Iterable[LayerKey], expected_steps: int, *, kind: PatchKind) -> None:
        source = self.condition if kind == "condition" else self.full
        for layer in layers:
            count = len(source.get(layer, []))
            if count != expected_steps:
                raise AssertionError(
                    f"Expected {expected_steps} cached {kind} activations for layer {layer}, found {count}"
                )


@dataclass
class ResidualPatchController:
    """Stateful record/inject controller for one sampler invocation."""

    expected_steps: int
    num_condition_frames: int
    record_condition_layers: tuple[LayerKey, ...] = ()
    record_full_layers: tuple[LayerKey, ...] = ()
    inject_bank: ActivationBank | None = None
    inject_layer: LayerKey | None = None
    inject_kind: PatchKind = "condition"
    bank: ActivationBank = field(default_factory=ActivationBank)
    step_index: int = 0
    _tokens_per_frame: int | None = None

    def condition_token_count(self, *, f: int, h: int, w: int) -> int:
        if self.num_condition_frames <= 0 or self.num_condition_frames >= f:
            raise AssertionError(
                f"num_condition_frames={self.num_condition_frames} is incompatible with token frames f={f}"
            )
        tokens_per_frame = h * w
        if self._tokens_per_frame is None:
            self._tokens_per_frame = tokens_per_frame
        elif self._tokens_per_frame != tokens_per_frame:
            raise AssertionError("Spatial token count changed across sampling steps")
        return self.num_condition_frames * tokens_per_frame

    def apply(self, x: torch.Tensor, *, layer: LayerKey, f: int, h: int, w: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected token tensor [B,N,D], got {tuple(x.shape)}")
        expected_tokens = f * h * w
        if x.shape[1] != expected_tokens:
            raise AssertionError(
                f"Token count {x.shape[1]} does not match f*h*w={expected_tokens}"
            )
        condition_count = self.condition_token_count(f=f, h=h, w=w)

        if layer in self.record_condition_layers:
            self.bank.record("condition", layer, x[:, :condition_count])
        if layer in self.record_full_layers:
            self.bank.record("full", layer, x)

        if self.inject_layer == layer:
            if self.inject_bank is None:
                raise AssertionError("inject_layer is set but inject_bank is missing")
            source = self.inject_bank.get(self.inject_kind, layer, self.step_index)
            source = source.to(device=x.device, dtype=x.dtype)
            if self.inject_kind == "condition":
                if tuple(source.shape) != tuple(x[:, :condition_count].shape):
                    raise AssertionError(
                        f"Condition cache shape {tuple(source.shape)} != receiver slice {tuple(x[:, :condition_count].shape)}"
                    )
                patched = x.clone()
                patched[:, :condition_count] = source
                x = patched
            else:
                if tuple(source.shape) != tuple(x.shape):
                    raise AssertionError(
                        f"Full cache shape {tuple(source.shape)} != receiver tensor {tuple(x.shape)}"
                    )
                x = source.clone()
        return x

    def finish_model_call(self) -> None:
        self.step_index += 1
        if self.step_index > self.expected_steps:
            raise AssertionError(
                f"DiT was called more than expected {self.expected_steps} sampling steps"
            )

    def finish_run(self) -> None:
        if self.step_index != self.expected_steps:
            raise AssertionError(
                f"Expected {self.expected_steps} DiT calls, observed {self.step_index}"
            )
        self.bank.assert_steps(self.record_condition_layers, self.expected_steps, kind="condition")
        self.bank.assert_steps(self.record_full_layers, self.expected_steps, kind="full")


def model_fn_wan_video_with_patch(
    *,
    dit: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    controller: ResidualPatchController,
) -> torch.Tensor:
    """Spring Wan model function with a post-residual record/inject hook."""
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = latents.new_zeros((latents.shape[0], 1, dit.dim))

    x = dit.patchify(latents)
    f, h, w = x.shape[2:]
    # The Spring config freezes temporal patch size to 1.  f must therefore
    # equal the latent-frame count; otherwise the temporal-prefix slice would
    # not correspond to condition frames and the run must fail closed.
    if f != latents.shape[2]:
        raise AssertionError(
            f"Temporal patchification changed {latents.shape[2]} latent frames into f={f}; "
            "condition-token prefix patching is not valid for this model"
        )
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
    freqs = torch.cat(
        [
            dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1).to(x.device)

    x = controller.apply(x, layer=-1, f=f, h=h, w=w)
    for layer_index, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs)
        x = controller.apply(x, layer=layer_index, f=f, h=h, w=w)
    x = dit.head(x, t)
    controller.finish_model_call()
    return dit.unpatchify(x, (f, h, w))


@torch.no_grad()
def sample_final_latents(
    *,
    pipe: Any,
    condition_latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    num_condition_frames: int,
    num_inference_steps: int,
    seed: int,
    controller: ResidualPatchController,
    sigma_shift: float = 5.0,
    denoising_strength: float = 1.0,
) -> torch.Tensor:
    """Latent-only sampler matching the Spring Wan pipeline's 20-step loop."""
    pipe.train(False)
    if condition_latents.shape[0] != 1:
        raise ValueError("Patching runner currently requires batch size 1")
    if condition_latents.shape[2] != num_condition_frames:
        raise ValueError(
            f"Condition has {condition_latents.shape[2]} latent frames; expected {num_condition_frames}"
        )
    if controller.expected_steps != num_inference_steps:
        raise AssertionError("Controller and sampler step counts differ")

    scheduler_was_training = bool(getattr(pipe.scheduler, "training", False))
    scheduler_shift = getattr(pipe.scheduler, "shift", None)
    pipe.scheduler.set_timesteps(
        num_inference_steps,
        denoising_strength=denoising_strength,
        shift=sigma_shift,
    )

    latent_frames = num_frames if pipe.vae.__class__.__name__ == "IdentityWanVideoVAE" else (num_frames - 1) // 4 + 1
    shape = (
        1,
        pipe.vae.z_dim,
        latent_frames,
        height // pipe.vae.upsampling_factor,
        width // pipe.vae.upsampling_factor,
    )
    latents = pipe.generate_noise(shape, seed=seed, rand_device=pipe.device).to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    condition_latents = condition_latents.to(device=pipe.device, dtype=pipe.torch_dtype)

    for progress_id, timestep_value in enumerate(pipe.scheduler.timesteps):
        timestep = timestep_value.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        latents[:, :, :num_condition_frames] = condition_latents
        velocity = model_fn_wan_video_with_patch(
            dit=pipe.dit,
            latents=latents,
            timestep=timestep,
            controller=controller,
        )
        latents = pipe.scheduler.step(velocity, timestep_value, latents)
        latents[:, :, :num_condition_frames] = condition_latents

    controller.finish_run()
    result = latents.detach().clone()

    if scheduler_was_training:
        kwargs: dict[str, Any] = {"training": True}
        if scheduler_shift is not None:
            kwargs["shift"] = scheduler_shift
        pipe.scheduler.set_timesteps(pipe.scheduler.num_train_timesteps, **kwargs)
    return result


def load_position_probe(path: Path, *, expected_history: str = "long") -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        data = {key: payload[key] for key in payload.files}
    if str(data["history"][0]) != expected_history:
        raise ValueError(f"Probe history is not {expected_history}")
    if tuple(int(value) for value in data["future_latent_range"]) != (
        FUTURE_LATENT_START,
        FUTURE_LATENT_STOP,
    ):
        raise ValueError("Probe future-latent convention mismatch")
    if str(data["feature_layout"][0]) != FEATURE_LAYOUT:
        raise ValueError("Probe feature-layout convention mismatch")
    if str(data["target"][0]) != TARGET_DESCRIPTION:
        raise ValueError("Probe target convention mismatch")
    coef = np.asarray(data["coef"], dtype=np.float32)
    if coef.shape != (4096,):
        raise ValueError(f"Unexpected probe coefficient shape: {coef.shape}")
    return {
        "coef": coef,
        "intercept": float(np.asarray(data["intercept"]).reshape(-1)[0]),
        "channel_mean": np.asarray(data["channel_mean"], dtype=np.float32),
        "channel_std": np.asarray(data["channel_std"], dtype=np.float32),
    }


def probe_positions(final_latents: torch.Tensor, probe: dict[str, Any]) -> np.ndarray:
    expected = (1, 16, 33, 16, 16)
    if tuple(final_latents.shape) != expected:
        raise ValueError(f"Expected final latent shape {expected}, got {tuple(final_latents.shape)}")
    future = (
        final_latents[0, :, FUTURE_LATENT_START:FUTURE_LATENT_STOP]
        .permute(1, 0, 2, 3)
        .float()
        .cpu()
        .numpy()
    )
    design = normalize_and_flatten(future, probe["channel_mean"], probe["channel_std"])
    values = design @ probe["coef"] + probe["intercept"]
    if values.shape != (16,) or not np.isfinite(values).all():
        raise AssertionError("P1 probe did not produce 16 finite positions")
    return values.astype(np.float64)


def fit_probe_omega(
    x_px: np.ndarray,
    *,
    slow_band: Any,
    fast_band: Any,
    tolerance: float,
) -> dict[str, Any]:
    values = np.asarray(x_px, dtype=np.float64)
    if values.shape != (16,) or not np.isfinite(values).all():
        raise ValueError("Expected 16 finite P1 positions")
    omega, rmse_px, center_offset_px, amplitude_px = fit_free_shm_frequency(
        values,
        np.ones(16, dtype=bool),
        fps=5,
        omega_low=max(0.2, slow_band.low * 0.55),
        omega_high=fast_band.high * 1.35,
    )
    omega_class = classify_omega(
        omega,
        slow_band,
        fast_band,
        tolerance=tolerance,
    )
    return {
        "omega": float(omega),
        "omega_class": str(omega_class),
        "fit_rmse_px": float(rmse_px),
        "fit_center_offset_px": float(center_offset_px),
        "fit_amplitude_px": float(amplitude_px),
        "x_px": [float(value) for value in values],
    }


def recovery_index(*, patched: float, conflict: float, aligned: float) -> float:
    denominator = aligned - conflict
    if not np.isfinite(denominator) or abs(denominator) < 1e-8:
        raise ValueError("Aligned/conflict frequency denominator is zero or non-finite")
    return float((patched - conflict) / denominator)


def max_abs_latent_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Latent shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}")
    return float((left.detach().float().cpu() - right.detach().float().cpu()).abs().max().item())
