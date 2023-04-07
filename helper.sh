#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=ddp-2way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --constraint=gmem48G

gpustat
date
echo “Job started.”
time torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --model DiT-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --sig transfered_plain
echo “Job completed.”