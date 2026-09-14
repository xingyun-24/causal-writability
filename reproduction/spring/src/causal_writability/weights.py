"""Install a model from an explicit file/URL or a published registry entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install(source, destination, expected=None):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
    os.close(fd)
    temp = Path(name)
    try:
        if source.startswith(('https://', 'http://')):
            with urllib.request.urlopen(source, timeout=60) as response, temp.open('wb') as out:
                shutil.copyfileobj(response, out)
        else:
            shutil.copyfile(source, temp)
        actual = digest(temp)
        if expected and actual != expected:
            raise ValueError('Downloaded file does not match the model registry')
        os.replace(temp, destination)
        return actual
    finally:
        temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', type=Path, default=Path('models/registry.json'))
    p.add_argument('--model')
    p.add_argument('--source', help='Explicit local checkpoint or HTTPS URL')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    entry = None
    if args.model:
        matches = [r for r in json.loads(args.registry.read_text()) if r['id'] == args.model]
        if len(matches) != 1:
            p.error('Unknown or ambiguous model ID')
        entry = matches[0]
    source = args.source or (entry and entry.get('url'))
    if not args.source and entry and entry.get('hf_repo'):
        from huggingface_hub import hf_hub_download
        source = hf_hub_download(repo_id=entry['hf_repo'], filename=entry['hf_filename'])
    if not source:
        p.error('This model has no published URL yet; pass --source with the supplied file or URL')
    install(source, args.out, entry and entry.get('sha256'))
    print(f'Model installed: {args.out}')


if __name__ == '__main__':
    main()
