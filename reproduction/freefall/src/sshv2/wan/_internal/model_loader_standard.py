"""Model loader for the standard Wan compatibility profile."""

import os, torch
from typing import TYPE_CHECKING, Type
from dataclasses import dataclass, field
from safetensors.torch import load_file as load_safetensors

if TYPE_CHECKING:
    from diffsynth.models import ModelManager


@dataclass
class CustomModelConfig:
    type: Type
    name: str
    kwargs: dict[str, any] = field(default_factory=dict)
    file_path: str | os.PathLike | None = None
    device: str | None = None
    torch_dtype: torch.dtype | None = None

    def load_model(self, model_manager: "ModelManager"):
        device = self.device or model_manager.device
        dtype  = self.torch_dtype or model_manager.torch_dtype
        
        model = self.type(**self.kwargs).to(device=device, dtype=dtype)
        if self.file_path is not None:
            print(f"Loading models from: {self.file_path}")

            ext = os.path.splitext(self.file_path)[1]
            if ext == ".safetensors":
                state_dict = load_safetensors(self.file_path)
            else:
                state_dict = torch.load(self.file_path)

            model.load_state_dict(state_dict)
            
        else:
            print(f"Initializing models")
            
        model_manager.model.append(model)
        model_manager.model_name.append(self.name)
        model_manager.model_path.append(self.file_path)

        print(f"    model_name: {self.name} model_class: {self.type.__name__}")
        print(f"    The following models are loaded: {[self.name]}.")
        return model
