#!/usr/bin/env python3
"""Launch one frozen Projectile Gravity V1 training run on the remote host."""
from __future__ import annotations

import argparse
import shlex
import subprocess


ROOT = "/data/home/yuanman/spring-mechanism-run/projectile-gravity-continuous-v2"
REPO = "/data/home/yuanman/physics-shortcuts-benchmarks"
PROFILES = {
    "short": {
        "config": "Train-short-v1.yaml",
        "latents": "train_short_v1",
        "run": "short-v1",
    },
    "long": {
        "config": "Train-long-v1.yaml",
        "latents": "train",
        "run": "long-v1",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", required=True, help="Remote CUDA device index")
    parser.add_argument("--host", default="yuanman@10.234.161.2")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    latent_dir = f"{ROOT}/data/latents/{profile['latents']}"
    config = f"{ROOT}/config/{profile['config']}"
    run = profile["run"]
    resolved = f"{ROOT}/training/{run}/resolved"
    checkpoints = f"{ROOT}/checkpoints/{run}"
    outputs = f"{ROOT}/output/{run}"
    log = f"{ROOT}/logs/train_{run}.log"
    worker = "\n".join((
        "set -euo pipefail",
        f"test $(find {shlex.quote(latent_dir)} -maxdepth 1 -name '*.pt' | wc -l) -eq 2048",
        f"test -s {shlex.quote(latent_dir + '/metadata.csv')}",
        f"test -s {shlex.quote(config)}",
        f"mkdir -p {shlex.quote(REPO + '/data')} {shlex.quote(resolved)} {shlex.quote(checkpoints)} {shlex.quote(outputs)} {shlex.quote(ROOT + '/logs')}",
        f"ln -sfn {shlex.quote(ROOT + '/data')} {shlex.quote(REPO + '/data/projectile_gravity_continuous_v2')}",
        f"cd {shlex.quote(REPO)}",
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)}",
        f"export PYTHONPATH={shlex.quote(REPO + '/src:' + REPO + '/lib/diffsynth')}",
        f"python3 {shlex.quote(ROOT + '/scripts/train_projectile.py')} --config {shlex.quote(config)} --resolved-dir {shlex.quote(resolved)} --ckpt-dir {shlex.quote(checkpoints)} --out-dir {shlex.quote(outputs)} --no-wandb",
    ))
    command = f"nohup bash -lc {shlex.quote(worker)} > {shlex.quote(log)} 2>&1 < /dev/null & echo $!"
    pid = subprocess.check_output(["ssh", args.host, command], text=True).strip()
    print(f"profile={args.profile} pid={pid} log={log}")


if __name__ == "__main__":
    main()
