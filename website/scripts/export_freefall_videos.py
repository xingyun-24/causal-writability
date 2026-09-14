"""Export the current paper's held-out Free Fall examples for browser playback."""
import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames-dir', type=Path, default=Path('website/references/freefall-frames'))
    p.add_argument('--out', type=Path, default=Path('website/public/videos/freefall'))
    a = p.parse_args()
    metadata = json.loads((a.frames_dir / 'examples.json').read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    videos = []
    for entry in metadata['examples']:
        frames = np.load(a.frames_dir / entry['frames'], allow_pickle=False)['frames']
        assert frames.shape == (64, 128, 128, 3) and frames.dtype == np.uint8
        direction = entry['pair_id'].rsplit('_', 1)[-1]
        condition = 'natural-conflict' if entry['condition'] == 'natural_conflict' else 'controller-edit'
        name = f'freefall-{direction}-{condition}'
        video = a.out / (name + '.mp4')
        temporary = video.with_suffix('.partial.mp4')
        command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '128x128',
                   '-framerate', str(metadata['fps']), '-i', 'pipe:0', '-an',
                   '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(temporary)]
        subprocess.run(command, input=frames.tobytes(), check=True)
        temporary.replace(video)
        Image.fromarray(frames[0]).save(a.out / (name + '.png'))
        videos.append({**entry, 'path': '/videos/freefall/' + video.name,
                       'poster': '/videos/freefall/' + name + '.png',
                       'n_frames': 64, 'fps': 20, 'width': 128, 'height': 128})
    (a.out / 'examples.json').write_text(json.dumps({
        'source_package': metadata['source_package'], 'paper_figure': 21,
        'checkpoint': 'Free Fall hist32 100K', 'intervention_block_zero_based': 1,
        'basis': 'current train-only PCA / directional [1,g_target] controller',
        'scope': 'two representative held-out pairs, not population success rates',
        'processing': 'H.264 yuv420p browser encoding only; unchanged frame order, FPS and spatial dimensions; no crop, recoloring, sharpening or interpolation. Reported metrics refer to original RGB frames.',
        'videos': videos}, indent=2) + '\n')
    print('Exported four current Free Fall videos and posters')


if __name__ == '__main__':
    main()
