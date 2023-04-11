#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --partition=ddp-2way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --constraint=gmem48G

gpustat
date
echo “Job started.”

#coupled:
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --model DiT-S/8     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --sig DiT-nosample --wandb 
torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --model DiT-DEQ-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --sig DEQ-nosample --wandb 


# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --model DiT-S/8     --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --sig DiT-sample --wandb --sample --sample-size 4
# torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py --model DiT-DEQ-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --sig DEQ-sample --wandb --sample --sample-size 4
# torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --model DiT-DEQ-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --sig DEQ-nosample --wandb --sample --sample-size 4

echo “Job completed.”