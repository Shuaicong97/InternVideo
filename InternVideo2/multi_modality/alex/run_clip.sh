#!/bin/bash -l

#SBATCH --job-name=run_demo_feat_extraction_clip
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:8
#SBATCH --output=/home/atuin/v100dd/v100dd19/sbatch/result-%x-%j.txt
#SBATCH -C a100_80

set -x

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 ../demo_feat_extraction_clip.py