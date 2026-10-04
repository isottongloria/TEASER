"""Per-frame face regions in image pixels, from a clip's SMPL-X fit (TEMPORAL_README.md 5.4).

Runs in RGB2SMPLX's env:

    PYTHONPATH=/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX ~/miniconda3/envs/rgb2smplx/bin/python \
        tools_temporal/export_face_regions.py --fits_of fits.tsv --clips list.txt --out <dir>

Writes ``<out>/<clip>.regions.npz``: ``mouth`` (T, K, 2) and ``eyes`` (T, K, 2),
the FLAME lips / eye-region vertices (src/temporal/flame_regions.py) of the
fitted mesh, projected with the fit's own camera -- exactly the regions
compute_real_occlusion.py measures real occlusion on. Synthetic occlusion is
placed and measured on the convex hulls of these, so real and synthetic
coverage are the same quantity.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

RGB2SMPLX = Path("/leonardo_work/IscrC_SLPSCALE/RGB2SMPLX")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits_of", type=Path, required=True, help="clip<TAB>fit npz path")
    parser.add_argument("--clips", type=Path, required=True, help="clip names (first column)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(RGB2SMPLX / "experiments/occlusion_protocols_smplx"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smplx_geometry as geometry
    from compute_real_occlusion import fit_for_geometry, flame_region_ids_in_smplx

    fit_of = dict(l.split("\t")[:2] for l in args.fits_of.read_text().splitlines() if l.strip())
    regions = flame_region_ids_in_smplx()
    args.out.mkdir(parents=True, exist_ok=True)
    done = 0
    for line in args.clips.read_text().splitlines():
        if not line.strip():
            continue
        clip = line.split("\t")[0]
        target = args.out / f"{clip}.regions.npz"
        if target.is_file():
            continue
        npz = fit_for_geometry(dict(np.load(fit_of[clip])), geometry)
        projected = geometry.project_all_frames(geometry.forward_vertices(npz), npz)
        np.savez(target, mouth=projected[:, regions["mouth"]].astype(np.float32),
                 eyes=projected[:, regions["eyes"]].astype(np.float32))
        done += 1
    print(f"[regions] {done} clips -> {args.out}")


if __name__ == "__main__":
    main()
