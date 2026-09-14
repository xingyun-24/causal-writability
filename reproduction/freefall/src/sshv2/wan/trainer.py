"""Public Wan training API."""

from sshv2.wan._internal.dataset import (
    LoadTensor,
    LoadVideoAsTensor,
    video_as_tensor_operator,
)
from sshv2.wan._internal.train_compat import train as train_compat
from sshv2.wan._internal.train_standard import train as train_standard
from sshv2.wan._internal.trainer_compat import (
    CompatibilityWanTrainingModule,
)
from sshv2.wan._internal.trainer_standard import WanTrainingModule
from sshv2.wan._internal.training_utils import (
    DiffusionTrainingModule,
    ModelLogger,
    launch_training_task,
)

__all__ = [
    "CompatibilityWanTrainingModule",
    "DiffusionTrainingModule",
    "LoadTensor",
    "LoadVideoAsTensor",
    "ModelLogger",
    "WanTrainingModule",
    "launch_training_task",
    "train_compat",
    "train_standard",
    "video_as_tensor_operator",
]
