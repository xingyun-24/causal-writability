"""Public Wan VAE and dataset-encoding API."""

from sshv2.wan._internal.encoding import (
    EncodeDatasetConfig,
    encode_dataset,
)
from sshv2.wan._internal.vae import (
    IdentityWanVideoVAE,
    WanVideoVAE,
)

__all__ = [
    "EncodeDatasetConfig",
    "IdentityWanVideoVAE",
    "WanVideoVAE",
    "encode_dataset",
]
