"""Training schema for the standard Wan path."""

from sshv2.wan._internal.wan_config import WanConfig
from sshv2.wan._internal.training_config import TrainingConfig


class StandardTrainingConfig(TrainingConfig):
    """Configuration accepted by the standard trainer."""

    model: WanConfig
