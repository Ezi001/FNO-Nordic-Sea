#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=nn8104k
#SBATCH --partition=accel
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gpus=2

#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

module purge
module load NRIS/GPU
module load hpc-container-wrapper

export PYTHONNOUSERSITE=1

PROJECT=/cluster/work/projects/nn8104k/esther_thesis/FNO-Nordic-Sea
# Use your wrapper
export PATH=/cluster/work/projects/nn8104k/esther_thesis/envs/cfo_gpu/bin:$PATH

cd $PROJECT

# Diagnostics (very useful)
echo "Python:"
which python
python --version

echo "NumPy:"
python -c "import numpy; print(numpy.__version__)"


python scripts/preprocess.py