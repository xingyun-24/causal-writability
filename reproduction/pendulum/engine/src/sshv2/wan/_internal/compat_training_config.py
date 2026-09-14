"""Training schema for checkpoints created by the compatibility Wan path."""

from sshv2.wan._internal.wan_compat_config import WanConfig
from sshv2.wan._internal.training_config import DataConfig, TrainingConfig


class CompatibilityTrainingConfig(TrainingConfig):
    """Configuration accepted by the compatibility trainer."""

    model: WanConfig | None = None
    data: DataConfig | None = None
    val_data: DataConfig | None = None
    benchmark: str = "wan_compat"
