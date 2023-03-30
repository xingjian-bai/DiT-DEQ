#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=ddp-4way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

gpustat
date
echo “Job started.”
time torchrun --nnodes=1 --nproc_per_node=4 deq_train.py \
  --sample-step 1000 --sample-size 64 --global-batch-size 128\
  --epochs 2000 --model DiT-S/8 \
  --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --image-size 256 \
  --num-classes 1000 \
  --log-every 100 --ckpt-every 1000 \
#   --wandb
echo “Job completed.”