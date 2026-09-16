#!/bin/bash
#SBATCH --job-name=teaser_gpu_smoke
#SBATCH --account=IscrC_SLPSCALE
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/leonardo_work/IscrC_SLPSCALE/TEASER/logs/gpu_smoke_test.%j.out

set -e
source /leonardo/home/userexternal/gisotto1/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_work/IscrC_SLPSCALE/TEASER/.conda_envs/teaser

cd /leonardo_work/IscrC_SLPSCALE/TEASER

python - <<'EOF'
import torch
print("torch:", torch.__version__, "cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    x = torch.randn(4, 4, device="cuda")
    y = x @ x
    print("matmul on GPU ok:", y.shape, y.device)

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
faces = torch.tensor([[0, 1, 2]], device="cuda")
meshes = Meshes(verts=[verts], faces=[faces])
pix_to_face, zbuf, bary, dists = rasterize_meshes(meshes, image_size=64)
print("pytorch3d CUDA rasterization ok:", pix_to_face.shape)
print("ALL GOOD")
EOF
