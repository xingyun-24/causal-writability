"""Public Wan denoiser API."""

from sshv2.wan._internal.model_compat import (
    WanModel as CompatibilityWanModel,
)
from sshv2.wan._internal.model_standard import WanModel

__all__ = ["CompatibilityWanModel", "WanModel"]
