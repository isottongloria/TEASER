"""Cut How2Sign raw videos into short chunks for the pilot (TEMPORAL_README.md 4.2d).

    python tools_temporal/make_how2sign_chunks.py --videos <dir> --list videos_used.txt --out <chunk dir> \
        --per_video 14 --seconds 4 --skip 5

How2Sign raw videos are 2-3 minutes long; the full RGB2SMPLX pipeline (needed
for the occlusion meter and the hand masks) would cost too much on whole
videos. Each video gives ``--per_video`` chunks of ``--seconds``, evenly spaced
between ``--skip`` seconds from the start and from the end. Chunks are
re-encoded (libx264, crf 12, native frame rate, no audio) and named
``h2s_<video id>_cNN`` (ASCII, never starting with '-'). Writes
``<out>/chunks.tsv``: chunk, source video, start second, fps.
"""

import argparse
import subprocess
from pathlib import Path

import cv2

FFMPEG = Path.home() / "miniconda3/envs/rgb2smplx-vid2smplx/bin/ffmpeg"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per_video", type=int, default=14)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--skip", type=float, default=5.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in [l.strip() for l in args.list.read_text().splitlines() if l.strip()]:
        video = args.videos / name
        cap = cv2.VideoCapture(str(video))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = frames / fps
        first, last = args.skip, duration - args.skip - args.seconds
        step = (last - first) / max(args.per_video - 1, 1)
        vid = name.replace("-rgb_front.mp4", "").replace("-", "m")
        for k in range(args.per_video):
            start = first + k * step
            chunk = f"h2s_{vid}_c{k:02d}"
            target = args.out / f"{chunk}.mp4"
            if not target.is_file():
                subprocess.run([str(FFMPEG), "-nostdin", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
                                "-i", str(video), "-t", f"{args.seconds:.3f}", "-an", "-c:v", "libx264",
                                "-crf", "12", "-pix_fmt", "yuv420p", str(target)], check=True)
            rows.append(f"{chunk}\t{name}\t{start:.3f}\t{fps:.3f}")
        print(f"[chunks] {name}: {fps:.2f} fps, {duration:.0f} s -> {args.per_video} chunks")
    (args.out / "chunks.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
