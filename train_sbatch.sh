#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --partition=ddp-4way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
# SBATCH --constraint=gmem48G

gpustat
date
echo “Job started.”
time torchrun --nnodes=1 --nproc_per_node=4 train.py \
  --sample-step 1000 --sample-size 16 \
  --epochs 2000 --model DiT-XL/2 \
  --data-path /work/lukemk/machine-learning-datasets/image-generation/ffhq-resized/resized/ --image-size 256 \
  --num-classes 1 \
  --log-every 100 --ckpt-every 1000 \
  --wandb
echo “Job completed.”

