"""Run a frozen Pendulum receiver through natural and full-matched generation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine/src"))
sys.path.insert(0, str(HERE / "engine/scripts/sshv2"))
from sshv2.experiments.pendulum.data import Appearance, PendulumParameters, config_from_mapping, pendulum_trajectory, render_video, write_video
from sshv2.experiments.pendulum.mechanism_pca import _condition_latents, _denoise, _load_runtime, _write_future_video
from reevaluate_pendulum_detection import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--history", choices=("short", "long"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (HERE / "project-page/data/strict_bank_128.jsonl").read_text().splitlines() if line.strip()]
    row = next(r for r in records if r["receiver_id"] == "B_082")
    config_path = HERE / "engine/configs/pendulum/frequency_color_circle_frequency_scan.yaml"
    config = config_from_mapping(yaml.safe_load(config_path.read_text())["data"])
    source = HERE / f"engine/configs/pendulum/Train-frequency_color_circle-large-{args.history}-50k.yaml"
    training = yaml.safe_load(source.read_text())
    training["model"]["vae"]["ckpt_file"] = str(args.vae.resolve())
    local_config = args.out / "runtime.yaml"
    local_config.write_text(yaml.safe_dump(training, sort_keys=False))
    runtime = SimpleNamespace(experiment_config=config_path, training_config=local_config,
        model_name="frequency_color_circle", history=args.history, checkpoint=args.checkpoint.resolve(),
        device=args.device, steps=20, block_index=12, condition_tokens=1088, hidden_size=1152)
    pipe, config = _load_runtime(runtime, [row])
    theta, _ = pendulum_trajectory(PendulumParameters(float(row["omega_true"]), float(row["amplitude_true"]), float(row["phase"])), fps=config.render.fps, num_frames=config.render.num_frames)
    conditions, activations, videos = {}, {}, {}
    for condition in ("aligned", "conflict"):
        input_path = args.out / f"{condition}-input.mp4"
        write_video(input_path, render_video(theta, Appearance(row[condition + "_color"], row[condition + "_shape"]), config.render), config.render.fps)
        conditions[condition] = _condition_latents(pipe, config, input_path, history=args.history)
        latent, activations[condition] = _denoise(pipe, config, conditions[condition], seed=int(row["generation_seed"]), steps=20, block_index=12, condition_tokens=1088, capture=True)
        assert activations[condition].shape == (20, 1088, 1152)
        videos[condition] = args.out / f"{condition}.mp4"
        _write_future_video(pipe, config, latent, videos[condition])
    edit = torch.from_numpy(activations["aligned"] - activations["conflict"]).to(device=pipe.device, dtype=pipe.torch_dtype)
    latent, _ = _denoise(pipe, config, conditions["conflict"], seed=int(row["generation_seed"]), steps=20, block_index=12, condition_tokens=1088, edit=edit)
    videos["full_matched"] = args.out / "full-matched.mp4"
    _write_future_video(pipe, config, latent, videos["full_matched"])
    contract = json.loads((HERE / "project-page/data/decoded_recovery_summary.json").read_text())["evaluator_contract"]
    measurements = {name: measure(path, row, config, **contract) for name, path in videos.items()}
    assert all(np.isfinite(r["omega_hat"]) and r["fit_valid"] for r in measurements.values())
    ga, gc, ge = [measurements[key]["omega_hat"] for key in ("aligned", "conflict", "full_matched")]
    result = {"history": args.history, "receiver_id": row["receiver_id"], "calls": 20,
              "block": 12, "measurements": measurements,
              "normalized_recovery": float((ge-gc)/(ga-gc)) if abs(ga-gc)>1e-8 else None,
              "scope": "runtime smoke test; not a full held-out Top-4 reproduction"}
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "measurements"}),flush=True)


if __name__ == "__main__":
    main()
