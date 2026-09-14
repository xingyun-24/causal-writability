#!/usr/bin/env python3
"""Create a reproducible pilot configuration for one observation window."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--base",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--window",type=int,required=True); p.add_argument("--steps",type=int,default=3000); a=p.parse_args()
    c=yaml.safe_load(a.base.read_text()); key=f"window_{a.window:02d}"; c["data"]["dataset"]=f"data/projectile_gravity_continuous_v2/latents/{key}"; c["data"]["labels"]=f"data/projectile_gravity_continuous_v2/latents/{key}/metadata.csv"; c["loader"]["num_training_steps"]=a.steps; c["log"]["project"]="projectile-gravity-window-calibration"; c["log"]["name"]=key; c["log"]["save_at"]=[a.steps]; c["log"]["save_last_every"]=a.steps; c["log"]["ckpt_dir"]=f"runs/projectile_gravity_continuous_v2/{key}/ckpt"; c["log"]["out_dir"]=f"runs/projectile_gravity_continuous_v2/{key}/output"; a.out.write_text(yaml.safe_dump(c,sort_keys=False))
if __name__ == "__main__": main()
