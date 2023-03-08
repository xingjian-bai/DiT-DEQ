#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --partition=ddp-2way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
# SBATCH --constraint=gmem24G

gpustat
date
echo “Job started.”
time torchrun --nnodes=1 --nproc_per_node=2 sample_ddp.py \
  --model DiT-XL/2 \
  --num-fid-samples 50000
echo “Job completed.”

