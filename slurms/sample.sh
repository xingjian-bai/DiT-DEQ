#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:2
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gmem24G


gpustat
date
echo “Job started.”
time torchrun --nnodes=1 --nproc_per_node=2 sample_ddp.py \
  --model DiT-XL/2 \
  --num-fid-samples 4096 --cfg-scale 1.0 
echo “Job completed.”

