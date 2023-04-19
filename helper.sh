#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --partition=ddp-4way
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
# SBATCH --constraint=gmem48G

gpustat
date
echo “Job started.”

# #train a one-backprop deq
# torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --model DiT-DEQ-B/4-sim   --sig back1 --sample

#train a random-thres dep
torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --model DiT-DEQ-B/4-rand-sim --sig backprop-rand --sample



#coupled:
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --wandb --model DiT-S/8     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --global-seed 10 --data-seed 20 --sig original
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --wandb --model DiT-S/8     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --global-seed 10 --data-seed 30 --sig data_seed
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --wandb --model DiT-DEQ-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --global-seed 10 --data-seed 20 --sig DEQ
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --wandb --model DiT-S/8     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --global-seed 40 --data-seed 20 --sig global_seed 

# torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --wandb --model DiT-B/4     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --global-seed 10 --data-seed 10 --sig BIG
# torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --wandb --model DiT-DEQ-B/4     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --global-seed 10 --data-seed 10 --sig BIG --sample

# torchrun --nnodes=1 --nproc_per_node=4 eval.py \
#   --model DiT-B/4 \
#   --ckpt /work/xingjian/DiT/results/074-DiT-B-4/checkpoints/0050000.pt

# python fid.py \
#   --model DiT-B/4 \
#   --ckpt /work/xingjian/DiT/results/075-DiT-B-4/checkpoints/0050000.pt

# torchrun --nnodes=1 --nproc_per_node=4 eval.py \
#   --model DiT-DEQ-B/4 \
#   --ckpt /work/xingjian/DiT/results/075-DiT-DEQ-B-4/checkpoints/0150000.pt
# python fid.py \
#   --model DiT-DEQ-B/4 \
#   --ckpt /work/xingjian/DiT/results/075-DiT-DEQ-B-4/checkpoints/0150000.pt




echo “Job completed.”