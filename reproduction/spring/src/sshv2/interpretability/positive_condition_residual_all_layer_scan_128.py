"""Audited single-site condition-residual patching for the frozen 128-pair positive scan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


class ActivationBankLike(Protocol):
    def get(self, kind: str, layer: int, step: int) -> torch.Tensor: ...

SCAN_SITES: tuple[int, ...] = (-1, *range(30))
CONTROL_SITE = 13


def site_label(site: int) -> str:
    if site == -1:
        return "pre_block"
    if 0 <= site <= 29:
        return f"after_block_{site:02d}"
    raise ValueError(f"Unsupported site: {site}")


def site_key(site: int) -> str:
    if site == -1:
        return "site_pre_block"
    if 0 <= site <= 29:
        return f"site_after_block_{site:02d}"
    raise ValueError(f"Unsupported site: {site}")


@dataclass
class AuditedSingleSiteResidualPatchController:
    """Replace one condition-token residual site on every model call.

    The controller fails closed unless the target site is hit exactly once per
    Flow-Matching call. It records the donor/live receiver distance before the
    overwrite and verifies that only the condition-token prefix was changed.
    """

    expected_steps: int
    num_condition_frames: int
    inject_bank: ActivationBankLike
    inject_site: int
    require_source_difference_every_step: bool = True
    require_source_equal_receiver: bool = False
    step_index: int = 0
    _tokens_per_frame: int | None = None
    _hits_this_call: int = 0
    source_live_max_abs_diff_by_step: list[float] = field(default_factory=list)
    future_suffix_max_abs_diff_after_patch_by_step: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if self.inject_site not in SCAN_SITES:
            raise ValueError(f"inject_site={self.inject_site} is outside the frozen scan")
        if self.require_source_difference_every_step and self.require_source_equal_receiver:
            raise ValueError("Source cannot be required both different from and equal to receiver")

    def condition_token_count(self, *, f: int, h: int, w: int) -> int:
        if self.num_condition_frames <= 0 or self.num_condition_frames >= f:
            raise AssertionError(
                f"num_condition_frames={self.num_condition_frames} is incompatible with token frames f={f}"
            )
        tokens_per_frame = h * w
        if self._tokens_per_frame is None:
            self._tokens_per_frame = tokens_per_frame
        elif self._tokens_per_frame != tokens_per_frame:
            raise AssertionError("Spatial token count changed across model calls")
        return self.num_condition_frames * tokens_per_frame

    def apply(self, x: torch.Tensor, *, layer: int, f: int, h: int, w: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,N,D], got {tuple(x.shape)}")
        if x.shape[1] != f * h * w:
            raise AssertionError(f"Token count {x.shape[1]} does not match f*h*w={f*h*w}")
        if layer != self.inject_site:
            return x
        if self.step_index >= self.expected_steps:
            raise AssertionError("Patch site was hit after the expected number of model calls")
        self._hits_this_call += 1
        if self._hits_this_call != 1:
            raise AssertionError(
                f"Patch site {self.inject_site} was hit more than once in model call {self.step_index}"
            )

        condition_count = self.condition_token_count(f=f, h=h, w=w)
        source = self.inject_bank.get("condition", self.inject_site, self.step_index)
        source = source.to(device=x.device, dtype=x.dtype)
        live_condition = x[:, :condition_count]
        if tuple(source.shape) != tuple(live_condition.shape):
            raise AssertionError(
                f"Source shape {tuple(source.shape)} != live condition shape {tuple(live_condition.shape)}"
            )

        source_live_diff = float((source.float() - live_condition.float()).abs().max().item())
        if not torch.isfinite(torch.tensor(source_live_diff)):
            raise AssertionError("Non-finite donor/live residual difference")
        if self.require_source_difference_every_step and source_live_diff <= 0.0:
            raise AssertionError(
                f"Donor equals live receiver at site {self.inject_site}, step {self.step_index}"
            )
        if self.require_source_equal_receiver and source_live_diff != 0.0:
            raise AssertionError(
                f"Self-replay donor differs from live receiver at site {self.inject_site}, "
                f"step {self.step_index}: {source_live_diff}"
            )

        future_before = x[:, condition_count:].clone()
        patched = x.clone()
        patched[:, :condition_count] = source
        if not torch.equal(patched[:, :condition_count], source):
            raise AssertionError("Condition-prefix overwrite was not exact")
        future_diff = float((patched[:, condition_count:] - future_before).float().abs().max().item())
        if future_diff != 0.0:
            raise AssertionError(f"Future-token suffix changed directly at injection: {future_diff}")

        self.source_live_max_abs_diff_by_step.append(source_live_diff)
        self.future_suffix_max_abs_diff_after_patch_by_step.append(future_diff)
        return patched

    def finish_model_call(self) -> None:
        if self._hits_this_call != 1:
            raise AssertionError(
                f"Expected exactly one hit at site {self.inject_site} in model call {self.step_index}; "
                f"observed {self._hits_this_call}"
            )
        self.step_index += 1
        self._hits_this_call = 0
        if self.step_index > self.expected_steps:
            raise AssertionError("Observed more model calls than expected")

    def finish_run(self) -> None:
        if self.step_index != self.expected_steps:
            raise AssertionError(
                f"Expected {self.expected_steps} model calls, observed {self.step_index}"
            )
        if len(self.source_live_max_abs_diff_by_step) != self.expected_steps:
            raise AssertionError("Incomplete source/live audit")
        if len(self.future_suffix_max_abs_diff_after_patch_by_step) != self.expected_steps:
            raise AssertionError("Incomplete future-suffix audit")

    def audit_dict(self) -> dict[str, Any]:
        self.finish_run()
        return {
            "inject_site": self.inject_site,
            "inject_site_label": site_label(self.inject_site),
            "expected_steps": self.expected_steps,
            "observed_steps": self.step_index,
            "source_live_max_abs_diff_by_step": [
                float(value) for value in self.source_live_max_abs_diff_by_step
            ],
            "source_live_max_abs_diff_min": float(min(self.source_live_max_abs_diff_by_step)),
            "source_live_max_abs_diff_max": float(max(self.source_live_max_abs_diff_by_step)),
            "future_suffix_max_abs_diff_after_patch_by_step": [
                float(value) for value in self.future_suffix_max_abs_diff_after_patch_by_step
            ],
            "future_suffix_directly_unchanged": all(
                value == 0.0 for value in self.future_suffix_max_abs_diff_after_patch_by_step
            ),
            "require_source_difference_every_step": self.require_source_difference_every_step,
            "require_source_equal_receiver": self.require_source_equal_receiver,
            "status": "PASS",
        }


@dataclass
class AuditedSingleSiteLiveAdditiveResidualController:
    """Add a frozen natural donor-minus-conflict prefix residual to live state.

    This is intentionally distinct from ``AuditedSingleSiteResidualPatchController``:
    it never overwrites receiver tokens.  At the selected post-residual site on
    every FM call it applies ``x_prefix <- x_prefix + (A_nat - C_nat)`` and
    fails closed unless that site is reached exactly once per call.
    """

    expected_steps: int
    num_condition_frames: int
    donor_bank: ActivationBankLike
    conflict_bank: ActivationBankLike
    inject_site: int
    step_index: int = 0
    _tokens_per_frame: int | None = None
    _hits_this_call: int = 0
    direction_max_abs_by_step: list[float] = field(default_factory=list)
    live_change_max_abs_by_step: list[float] = field(default_factory=list)
    additive_assignment_max_abs_error_by_step: list[float] = field(default_factory=list)
    future_suffix_max_abs_diff_after_patch_by_step: list[float] = field(default_factory=list)
    initial_live_vs_natural_conflict_max_abs: float | None = None

    def __post_init__(self) -> None:
        if self.expected_steps <= 0:
            raise ValueError("expected_steps must be positive")
        if self.inject_site not in SCAN_SITES:
            raise ValueError(f"inject_site={self.inject_site} is outside the frozen scan")

    def condition_token_count(self, *, f: int, h: int, w: int) -> int:
        if not 0 < self.num_condition_frames < f:
            raise AssertionError("Invalid condition latent-frame count")
        tokens_per_frame = h * w
        if self._tokens_per_frame is None:
            self._tokens_per_frame = tokens_per_frame
        elif self._tokens_per_frame != tokens_per_frame:
            raise AssertionError("Spatial token count changed across model calls")
        return self.num_condition_frames * tokens_per_frame

    def apply(self, x: torch.Tensor, *, layer: int, f: int, h: int, w: int) -> torch.Tensor:
        if x.ndim != 3 or x.shape[0] != 1:
            raise AssertionError(f"Expected [1,N,D], got {tuple(x.shape)}")
        if x.shape[1] != f * h * w:
            raise AssertionError("Token geometry changed")
        if layer != self.inject_site:
            return x
        if not 0 <= self.step_index < self.expected_steps:
            raise AssertionError("Patch site was hit outside the expected FM calls")
        self._hits_this_call += 1
        if self._hits_this_call != 1:
            raise AssertionError(
                f"Patch site {self.inject_site} was hit more than once in model call {self.step_index}"
            )

        condition_count = self.condition_token_count(f=f, h=h, w=w)
        live_before = x[:, :condition_count]
        donor = self.donor_bank.get("condition", self.inject_site, self.step_index).to(
            device=x.device, dtype=x.dtype
        )
        conflict = self.conflict_bank.get("condition", self.inject_site, self.step_index).to(
            device=x.device, dtype=x.dtype
        )
        if tuple(donor.shape) != tuple(live_before.shape) or tuple(conflict.shape) != tuple(live_before.shape):
            raise AssertionError("Natural donor/conflict prefix shape differs from live receiver")
        direction = donor - conflict
        direction_max = float(direction.float().abs().max().item())
        if not torch.isfinite(direction).all() or direction_max <= 0.0:
            raise AssertionError("Natural donor-minus-conflict direction is non-finite or vacuous")
        if self.step_index == 0:
            initial_replay = float((live_before.float() - conflict.float()).abs().max().item())
            self.initial_live_vs_natural_conflict_max_abs = initial_replay
            if initial_replay != 0.0:
                raise AssertionError(
                    f"Initial conflict live state differs from its natural reference: {initial_replay}"
                )

        future_before = x[:, condition_count:].clone()
        expected_prefix = live_before + direction
        patched = x.clone()
        patched[:, :condition_count] = expected_prefix
        assignment_error = float(
            (patched[:, :condition_count].float() - expected_prefix.float()).abs().max().item()
        )
        suffix_error = float(
            (patched[:, condition_count:].float() - future_before.float()).abs().max().item()
        )
        if assignment_error != 0.0:
            raise AssertionError(f"Live additive assignment failed: {assignment_error}")
        if suffix_error != 0.0:
            raise AssertionError(f"Live additive edit changed future-token suffix: {suffix_error}")
        change = float((patched[:, :condition_count].float() - live_before.float()).abs().max().item())
        if change <= 0.0:
            raise AssertionError("Nonzero live-additive direction did not change condition prefix")

        self.direction_max_abs_by_step.append(direction_max)
        self.live_change_max_abs_by_step.append(change)
        self.additive_assignment_max_abs_error_by_step.append(assignment_error)
        self.future_suffix_max_abs_diff_after_patch_by_step.append(suffix_error)
        return patched

    def finish_model_call(self) -> None:
        if self._hits_this_call != 1:
            raise AssertionError(
                f"Expected exactly one hit at site {self.inject_site} in model call {self.step_index}; "
                f"observed {self._hits_this_call}"
            )
        self.step_index += 1
        self._hits_this_call = 0

    def finish_run(self) -> None:
        if self.step_index != self.expected_steps:
            raise AssertionError(f"Expected {self.expected_steps} model calls, observed {self.step_index}")
        for values in (
            self.direction_max_abs_by_step,
            self.live_change_max_abs_by_step,
            self.additive_assignment_max_abs_error_by_step,
            self.future_suffix_max_abs_diff_after_patch_by_step,
        ):
            if len(values) != self.expected_steps:
                raise AssertionError("Incomplete live-additive audit")

    def audit_dict(self) -> dict[str, Any]:
        self.finish_run()
        return {
            "controller": type(self).__name__,
            "operator": "live_prefix <- live_prefix + (natural_aligned_prefix - natural_conflict_prefix)",
            "inject_site": self.inject_site,
            "inject_site_label": site_label(self.inject_site),
            "expected_steps": self.expected_steps,
            "observed_steps": self.step_index,
            "condition_prefix_only": True,
            "replacement": False,
            "recorder": False,
            "initial_live_vs_natural_conflict_max_abs": self.initial_live_vs_natural_conflict_max_abs,
            "direction_max_abs_by_step": self.direction_max_abs_by_step,
            "live_change_max_abs_by_step": self.live_change_max_abs_by_step,
            "additive_assignment_max_abs_error_by_step": self.additive_assignment_max_abs_error_by_step,
            "future_suffix_max_abs_diff_after_patch_by_step": self.future_suffix_max_abs_diff_after_patch_by_step,
            "future_suffix_directly_unchanged": all(
                value == 0.0 for value in self.future_suffix_max_abs_diff_after_patch_by_step
            ),
            "status": "PASS",
        }


@dataclass
class AuditedSingleSiteLiveAdditiveDirectionController:
    """Apply a precomputed, per-step direction to a live receiver prefix.

    Used for held-out full-d and fit-PCA projected-d interventions.  Direction
    tensors are CPU snapshots supplied by the caller; this controller has no
    recording or replacement path.
    """

    expected_steps: int
    num_condition_frames: int
    direction_by_step: list[torch.Tensor]
    conflict_reference_by_step: list[torch.Tensor]
    inject_site: int
    arm: str
    step_index: int = 0
    _tokens_per_frame: int | None = None
    _hits_this_call: int = 0
    direction_max_abs_by_step: list[float] = field(default_factory=list)
    assignment_error_by_step: list[float] = field(default_factory=list)
    future_suffix_max_abs_diff_after_patch_by_step: list[float] = field(default_factory=list)
    initial_live_vs_conflict_max_abs: float | None = None

    def __post_init__(self) -> None:
        if self.inject_site not in SCAN_SITES:
            raise ValueError("Injection site is outside the supported model sites")
        if len(self.direction_by_step) != self.expected_steps or len(self.conflict_reference_by_step) != self.expected_steps:
            raise AssertionError("Direction and conflict-reference sequences must cover every FM call")

    def condition_token_count(self, *, f: int, h: int, w: int) -> int:
        if not 0 < self.num_condition_frames < f:
            raise AssertionError("Invalid condition-frame geometry")
        count = self.num_condition_frames * h * w
        if self._tokens_per_frame is None:
            self._tokens_per_frame = h * w
        elif self._tokens_per_frame != h * w:
            raise AssertionError("Spatial token count changed")
        return count

    def apply(self, x: torch.Tensor, *, layer: int, f: int, h: int, w: int) -> torch.Tensor:
        if layer != self.inject_site:
            return x
        if self.step_index >= self.expected_steps:
            raise AssertionError("Unexpected extra FM call")
        self._hits_this_call += 1
        if self._hits_this_call != 1:
            raise AssertionError("Injection site was hit more than once in one FM call")
        count = self.condition_token_count(f=f, h=h, w=w)
        live = x[:, :count]
        direction = self.direction_by_step[self.step_index].to(device=x.device, dtype=x.dtype)
        reference = self.conflict_reference_by_step[self.step_index].to(device=x.device, dtype=x.dtype)
        if direction.ndim == 2:
            direction = direction.unsqueeze(0)
        if reference.ndim == 2:
            reference = reference.unsqueeze(0)
        if tuple(direction.shape) != tuple(live.shape) or tuple(reference.shape) != tuple(live.shape):
            raise AssertionError("Direction/reference shape differs from live condition prefix")
        if not torch.isfinite(direction).all():
            raise AssertionError("Non-finite injection direction")
        direction_max = float(direction.float().abs().max().item())
        if direction_max <= 0.0:
            raise AssertionError("Vacuous injection direction")
        if self.step_index == 0:
            self.initial_live_vs_conflict_max_abs = float((live.float() - reference.float()).abs().max().item())
            if self.initial_live_vs_conflict_max_abs != 0.0:
                raise AssertionError("Initial live conflict state differs from its natural matched reference")
        future_before = x[:, count:].clone()
        expected = live + direction
        patched = x.clone()
        patched[:, :count] = expected
        assignment_error = float((patched[:, :count].float() - expected.float()).abs().max().item())
        suffix_error = float((patched[:, count:].float() - future_before.float()).abs().max().item())
        if assignment_error != 0.0 or suffix_error != 0.0:
            raise AssertionError("Live additive prefix invariant failed")
        self.direction_max_abs_by_step.append(direction_max)
        self.assignment_error_by_step.append(assignment_error)
        self.future_suffix_max_abs_diff_after_patch_by_step.append(suffix_error)
        return patched

    def finish_model_call(self) -> None:
        if self._hits_this_call != 1:
            raise AssertionError("Expected exactly one injection-site hit per FM call")
        self.step_index += 1
        self._hits_this_call = 0

    def finish_run(self) -> None:
        if self.step_index != self.expected_steps or any(len(x) != self.expected_steps for x in (self.direction_max_abs_by_step, self.assignment_error_by_step, self.future_suffix_max_abs_diff_after_patch_by_step)):
            raise AssertionError("Incomplete live-additive run")

    def audit_dict(self) -> dict[str, Any]:
        self.finish_run()
        return {
            "controller": type(self).__name__, "arm": self.arm,
            "operator": "live_prefix <- live_prefix + supplied_direction",
            "inject_site": self.inject_site, "inject_site_label": site_label(self.inject_site),
            "expected_steps": self.expected_steps, "observed_steps": self.step_index,
            "condition_prefix_only": True, "replacement": False, "recorder": False,
            "initial_live_vs_conflict_max_abs": self.initial_live_vs_conflict_max_abs,
            "direction_max_abs_by_step": self.direction_max_abs_by_step,
            "additive_assignment_max_abs_error_by_step": self.assignment_error_by_step,
            "future_suffix_max_abs_diff_after_patch_by_step": self.future_suffix_max_abs_diff_after_patch_by_step,
            "future_suffix_directly_unchanged": all(value == 0.0 for value in self.future_suffix_max_abs_diff_after_patch_by_step),
            "status": "PASS",
        }
