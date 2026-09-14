"""Condition-only Q/K/V hooks reused from the paper experiments."""
from dataclasses import dataclass, field
from typing import Any
import torch

@dataclass
class QKVBank:
    values: dict[str, list[torch.Tensor]] = field(
        default_factory=lambda: {"q": [], "k": [], "v": []}
    )


@dataclass
class ConditionQKVController:
    expected_steps: int
    condition_frames: int
    total_frames: int
    mode: str = "record"
    components: tuple[str, ...] = ()
    source: QKVBank | None = None
    active_steps: tuple[int, ...] | None = None
    head_indices: tuple[int, ...] | None = None
    num_heads: int | None = None
    # 1.0 is the historical exact source replacement.  Values in [0, 1]
    # interpolate from the live target to the recorded source; values > 1
    # extrapolate the same fit-frozen K/V direction.  Keeping the default at
    # 1.0 makes all existing QKV runs bit-for-bit protocol compatible.
    blend_alpha: float = 1.0
    allow_vacuous_steps: bool = False
    bank: QKVBank = field(default_factory=QKVBank)
    hits: dict[str, int] = field(default_factory=lambda: {"q": 0, "k": 0, "v": 0})
    max_source_live_difference: dict[str, list[float]] = field(
        default_factory=lambda: {"q": [], "k": [], "v": []}
    )
    handles: list[Any] = field(default_factory=list)

    def attach(self, block: Any) -> None:
        if self.handles:
            raise AssertionError("QKV controller already attached")
        for name in ("q", "k", "v"):
            module = getattr(block.self_attn, name)
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _hook(self, name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise AssertionError(f"Unexpected {name} output: {type(output)} {getattr(output, 'shape', None)}")
            index = self.hits[name]
            if index >= self.expected_steps:
                raise AssertionError(f"Too many {name} calls")
            if output.shape[1] % self.total_frames:
                raise AssertionError("Token count is not divisible by latent frame count")
            condition_count = self.condition_frames * (output.shape[1] // self.total_frames)
            live_condition = output[:, :condition_count]
            if self.mode == "record":
                self.bank.values[name].append(live_condition.detach().cpu().clone())
                result = output
            elif self.mode == "inject":
                if self.source is None:
                    raise AssertionError("Injection requires a source bank")
                step_is_active = self.active_steps is None or index in self.active_steps
                if name not in self.components or not step_is_active:
                    result = output
                else:
                    source = self.source.values[name][index].to(output.device, output.dtype)
                    if self.head_indices is not None:
                        if self.num_heads is None or output.shape[-1] % self.num_heads:
                            raise AssertionError("Head-selective patch requires a compatible num_heads")
                        head_dim = output.shape[-1] // self.num_heads
                        source_heads = source.view(source.shape[0], source.shape[1], self.num_heads, head_dim)
                        live_heads = live_condition.view(
                            live_condition.shape[0], live_condition.shape[1], self.num_heads, head_dim
                        )
                        difference = float(
                            (source_heads[:, :, list(self.head_indices)].float()
                             - live_heads[:, :, list(self.head_indices)].float()).abs().max()
                        )
                    else:
                        difference = float((source.float() - live_condition.float()).abs().max())
                    if difference <= 0 and not self.allow_vacuous_steps:
                        raise AssertionError(f"Vacuous {name} intervention at FM call {index}")
                    result = output.clone()
                    future_before = output[:, condition_count:].clone()
                    if self.head_indices is None:
                        if float(self.blend_alpha) == 1.0:
                            # Preserve the canonical exact-replacement
                            # semantics.  In bf16, live+(source-live) is not
                            # algebraically exact after intermediate rounding.
                            result[:, :condition_count] = source
                        elif float(self.blend_alpha) == 0.0:
                            result[:, :condition_count] = live_condition
                        else:
                            result[:, :condition_count] = (
                                live_condition
                                + float(self.blend_alpha) * (source - live_condition)
                            )
                        if float(self.blend_alpha) == 1.0 and not torch.equal(
                            result[:, :condition_count], source
                        ):
                            raise AssertionError("Exact full-component source replacement failed")
                    else:
                        result_condition_heads = result[:, :condition_count].view(
                            result.shape[0], condition_count, self.num_heads, head_dim
                        )
                        selected_live = live_heads[:, :, list(self.head_indices)]
                        selected_source = source_heads[:, :, list(self.head_indices)]
                        if float(self.blend_alpha) == 1.0:
                            result_condition_heads[:, :, list(self.head_indices)] = selected_source
                        elif float(self.blend_alpha) == 0.0:
                            result_condition_heads[:, :, list(self.head_indices)] = selected_live
                        else:
                            result_condition_heads[:, :, list(self.head_indices)] = (
                                selected_live
                                + float(self.blend_alpha) * (selected_source - selected_live)
                            )
                        if float(self.blend_alpha) == 1.0 and not torch.equal(
                            result_condition_heads[:, :, list(self.head_indices)], selected_source
                        ):
                            raise AssertionError("Exact head-selective source replacement failed")
                    if not torch.equal(result[:, condition_count:], future_before):
                        raise AssertionError(f"{name} intervention directly changed future tokens")
                    self.max_source_live_difference[name].append(difference)
            else:
                raise ValueError(self.mode)
            self.hits[name] += 1
            return result
        return hook

    def finish(self) -> None:
        if any(value != self.expected_steps for value in self.hits.values()):
            raise AssertionError(f"Incomplete QKV hooks: {self.hits}")
        if self.mode == "record":
            if any(len(self.bank.values[name]) != self.expected_steps for name in self.bank.values):
                raise AssertionError("Incomplete recorded QKV bank")
        else:
            for name in self.components:
                expected_active = self.expected_steps if self.active_steps is None else len(self.active_steps)
                if len(self.max_source_live_difference[name]) != expected_active:
                    raise AssertionError(f"Incomplete {name} injection audit")

