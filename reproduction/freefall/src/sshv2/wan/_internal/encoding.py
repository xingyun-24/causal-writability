"""Shared Wan VAE dataset encoding used by the thin encode CLI."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffsynth.models.model_manager import ModelManager
from diffsynth.trainers.unified_dataset import ImageCropAndResize
from tqdm import tqdm

from sshv2.wan._internal.wan_config import WanVAEConfig
from sshv2.wan._internal.dataset import LoadVideoAsTensor


@dataclass(frozen=True)
class EncodeDatasetConfig:
    vae: Path
    batch_size: int = 16
    device: str = "cuda"
    cpu_threads: int | None = None
    start: int = 0
    stop: int | None = None
    finalize_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["vae"] = str(self.vae)
        return value


def _read_rows(source: Path) -> tuple[list[dict[str, Any]], str]:
    legacy = source / "metadata.csv"
    if legacy.exists():
        with legacy.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle)), "metadata.csv"
    common = source / "samples.jsonl"
    if common.exists():
        rows = [
            json.loads(line)
            for line in common.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return rows, "samples.jsonl"
    raise FileNotFoundError(
        f"Expected metadata.csv or samples.jsonl in {source}"
    )


def _latent_relative(video: str) -> Path:
    return Path(video).with_suffix(".pt")


def _copy_sidecar(source: Path, output: Path, row: dict[str, Any]) -> None:
    metadata = row.get("metadata")
    if not isinstance(metadata, str):
        return
    source_path = source / metadata
    if not source_path.is_file():
        return
    destination = output / metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source_path, destination)


def _write_manifest(
    source: Path,
    output: Path,
    rows: list[dict[str, Any]],
    source_format: str,
    config: EncodeDatasetConfig,
) -> None:
    encoded = []
    for row in rows:
        item = dict(row)
        item["video"] = _latent_relative(str(item["video"])).as_posix()
        encoded.append(item)
        _copy_sidecar(source, output, row)
    if source_format == "metadata.csv":
        with (output / "metadata.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(encoded[0]),
            )
            writer.writeheader()
            writer.writerows(encoded)
    else:
        with (output / "samples.jsonl").open(
            "w",
            encoding="utf-8",
        ) as handle:
            for row in encoded:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        dataset_info = source / "dataset.json"
        if dataset_info.exists():
            shutil.copy2(dataset_info, output / "dataset.json")
    (output / "encoding_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "vae": str(config.vae),
                "samples": len(rows),
                "source_format": source_format,
                "shape": "[1,16,13,16,16]",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def encode_dataset(
    dataset_dir: Path,
    latent_dir: Path,
    config: EncodeDatasetConfig,
) -> Path:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.start < 0:
        raise ValueError("start must be non-negative")
    if config.cpu_threads is not None:
        torch.set_num_threads(config.cpu_threads)
    latent_dir.mkdir(parents=True, exist_ok=True)
    rows, source_format = _read_rows(dataset_dir)
    if not rows:
        raise ValueError(f"Dataset has no samples: {dataset_dir}")
    stop = (
        len(rows)
        if config.stop is None
        else min(config.stop, len(rows))
    )
    if stop < config.start:
        raise ValueError("stop must be greater than or equal to start")
    selected = rows[config.start:stop]
    todo = (
        []
        if config.finalize_only
        else [
            row
            for row in selected
            if not (
                latent_dir / _latent_relative(str(row["video"]))
            ).exists()
        ]
    )
    if todo:
        manager = ModelManager(
            device=config.device,
            torch_dtype=torch.bfloat16,
        )
        WanVAEConfig(
            z_dim=16,
            queued=False,
            ckpt_file=config.vae,
        ).get_model_config(
            torch_dtype=torch.bfloat16,
        ).load_model(manager)
        vae = manager.fetch_model("wan_video_vae").eval()
        loader = LoadVideoAsTensor(
            49,
            4,
            1,
            frame_processor=ImageCropAndResize(
                128,
                128,
                128 * 128,
                16,
                16,
            ),
        )
        for index in tqdm(
            range(0, len(todo), config.batch_size),
            desc=f"{dataset_dir}:{config.start}:{stop}",
        ):
            batch = todo[index:index + config.batch_size]
            pixels = torch.cat(
                [
                    loader(dataset_dir / str(row["video"]))
                    for row in batch
                ]
            ).to(config.device, dtype=torch.bfloat16)
            with torch.no_grad():
                latents = vae.encode(
                    pixels,
                    device=config.device,
                    tiled=False,
                ).cpu()
            for row, latent in zip(batch, latents):
                destination = (
                    latent_dir
                    / _latent_relative(str(row["video"]))
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(latent.unsqueeze(0), destination)
    if config.finalize_only:
        missing = [
            str(row["video"])
            for row in rows
            if not (
                latent_dir / _latent_relative(str(row["video"]))
            ).exists()
        ]
        if missing:
            raise RuntimeError(
                f"cannot finalize: {len(missing)} latent shards "
                f"are incomplete"
            )
    if config.finalize_only or (
        config.start == 0 and stop == len(rows)
    ):
        _write_manifest(
            dataset_dir,
            latent_dir,
            rows,
            source_format,
            config,
        )
    return latent_dir
