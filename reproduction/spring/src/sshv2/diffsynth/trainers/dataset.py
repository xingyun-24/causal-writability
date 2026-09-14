"""Dataset operators needed by the standalone spring experiment."""

from __future__ import annotations

import numpy as np
import torch
from einops import repeat
from typing_extensions import override

from diffsynth.trainers.unified_dataset import (
    DataProcessingOperator,
    ImageCropAndResize,
    LoadVideo,
    RouteByExtensionName,
    RouteByType,
    ToAbsolutePath,
)


class LoadVideoAsTensor(LoadVideo):
    @override
    def __call__(self, data):
        frames = super().__call__(data)
        return self.preprocess_video(frames)

    def preprocess_image(
        self,
        image,
        torch_dtype=None,
        device=None,
        pattern="B C H W",
        min_value=-1,
        max_value=1,
    ):
        image = torch.tensor(np.array(image, dtype=np.float32), device=device)
        image = (image * ((max_value - min_value) / 255.0) + min_value).to(dtype=torch_dtype)
        return repeat(image, f"H W C -> {pattern}", **({"B": 1} if "B" in pattern else {}))

    def preprocess_video(
        self,
        video,
        torch_dtype=None,
        device=None,
        pattern="B C T H W",
        min_value=-1,
        max_value=1,
    ):
        frames = [
            self.preprocess_image(
                image,
                torch_dtype=torch_dtype,
                device=device,
                min_value=min_value,
                max_value=max_value,
            )
            for image in video
        ]
        return torch.stack(frames, dim=pattern.index("T") // 2)


class LoadTensor(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location

    def __call__(self, data):
        try:
            return torch.load(data, map_location=self.map_location, weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load tensor file: {data}") from exc


def video_as_tensor_operator(
    base_path="",
    max_pixels=1920 * 1080,
    height=None,
    width=None,
    height_division_factor=16,
    width_division_factor=16,
    num_frames=81,
    time_division_factor=4,
    time_division_remainder=1,
    map_location="cpu",
):
    return RouteByType(
        operator_map=[
            (
                str,
                ToAbsolutePath(base_path)
                >> RouteByExtensionName(
                    operator_map=[
                        (
                            ("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
                            LoadVideoAsTensor(
                                num_frames,
                                time_division_factor,
                                time_division_remainder,
                                frame_processor=ImageCropAndResize(
                                    height,
                                    width,
                                    max_pixels,
                                    height_division_factor,
                                    width_division_factor,
                                ),
                            ),
                        ),
                        (("pt", "pth"), LoadTensor(map_location)),
                    ]
                ),
            ),
        ]
    )
