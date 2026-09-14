"""Restore Free Fall models and saved arrays at the runtime's expected paths."""
import argparse
import json
import shutil
from pathlib import Path


def restore(root, entry, repo_id, download):
    target = (root / entry["path"]).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Download target is outside the repository")
    cached = Path(download(repo_id=repo_id, filename=entry["filename"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-only", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    manifest = json.loads((args.root / "artifacts.json").read_text())
    entries = [(manifest["repo_id"], item) for item in manifest["files"]]
    if not args.artifacts_only:
        entries += [
            (manifest["repo_id"], {
                "filename": "freefall/freefall-hist32-100k.safetensors",
                "path": "reproduction/freefall/checkpoints/short-v3-large-fixedpos-freefall-hist32/hist32/step-100000.safetensors",
            }),
            ("Wan-AI/Wan2.1-T2V-1.3B", {
                "filename": "Wan2.1_VAE.pth",
                "path": "reproduction/freefall/models/Wan2.1_VAE.pth",
            }),
        ]
    for i, (repo_id, item) in enumerate(entries, 1):
        target = restore(args.root, item, repo_id, hf_hub_download)
        print(f"[{i}/{len(entries)}] {target.relative_to(args.root.resolve())}", flush=True)


if __name__ == "__main__":
    main()
