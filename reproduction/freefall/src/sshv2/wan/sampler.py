"""Public Wan prediction and video-interface API."""

from sshv2.wan._internal.interface import (
    infer_video_size,
    save_video,
    video_to_tensor,
)
from sshv2.wan._internal.sampler_compat import (
    WanVideoPipeline as CompatibilityWanVideoPipeline,
    model_fn_wan_video as compatibility_model_fn,
)
from sshv2.wan._internal.sampler_standard import (
    WanVideoPipeline,
    model_fn_wan_video,
)

__all__ = [
    "CompatibilityWanVideoPipeline",
    "WanVideoPipeline",
    "compatibility_model_fn",
    "infer_video_size",
    "model_fn_wan_video",
    "save_video",
    "video_to_tensor",
]
