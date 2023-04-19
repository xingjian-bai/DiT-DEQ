# torchrun  --nnodes=1 --nproc_per_node=2 train.py --sample-step 250 --sample-size 256 --epochs 2000 --model DiT-B/4 --data-path /work/xingjian/diffuser-self/flowers102/ --num-classes 1 --log-every 100 --ckpt-every 200 --image-size 64


# for flowers
torchrun --nnodes=1 --nproc_per_node=2 train.py \
  --sample-step 250 --sample-size 128 \
  --epochs 2000 --model DiT-B/4 \
  --data-path /work/xingjian/diffuser-self/flowers102/ --image-size 64 \
  --num-classes 1 \
  --log-every 500 --ckpt-every 2000 

# for ffhq
torchrun --nnodes=1 --nproc_per_node=2 train.py \
  --sample-step 250 --sample-size 128 \
  --epochs 2000 --model DiT-B/4 \
  --data-path /work/lukemk/machine-learning-datasets/image-generation/ffhq-resized/resized/ --image-size 256 \
  --num-classes 1 \
  --log-every 500 --ckpt-every 2000 

# for ffhq, deq
torchrun --nnodes=1 --nproc_per_node=4 deq_train.py \
  --sample-step 250 --sample-size 128 --global-batch-size 128\
  --epochs 2000 --model DiT-DEQ-B/4 \
  --data-path /work/lukemk/machine-learning-datasets/image-generation/ffhq-resized/resized/ --image-size 256 \
  --num-classes 1 \
  --log-every 100 --ckpt-every 1000 \
  --wandb


# imagenet
torchrun --nnodes=1 --nproc_per_node=1 deq_train.py \
  --sample-step 1000 --sample-size 8 --global-batch-size 128\
  --epochs 2000 --model DiT-DEQ-S/8\
  --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --image-size 256 \
  --num-classes 1000 \
  --log-every 100 --ckpt-every 1000 \
  --wandb
# DiT-DEQ-S/8

###################
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=ddp-4way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

gpustat
date
echo “Job started.”
time torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --model DiT-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/
echo “Job completed.”
###################


# eval
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp.py \
  --model DiT-XL/2 \
  --num-fid-samples 50000 --cfg-scale 1.0 


# srun
srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=gpu \
 --pty --gres=gpu:4 --constraint=gmem48G /bin/zsh

srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=ddp-4way \
 --pty --gres=gpu:4 --constraint=gmem48G /bin/zsh

srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=gpu \
 --pty --gres=gpu:2 --constraint=gmem48G /bin/zsh

srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=low-prio-gpu \
  --mem=100G --pty --gres=gpu:2 --constraint=gmem48G /bin/zsh

srun --ntasks=1 --time=24:00:00 --cpus-per-task=12 --partition=ddp-4way \
  --mem=50G --pty /bin/zsh


## NEW TRY
torchrun --nnodes=1 --nproc_per_node=1 deq_train.py \
  --global-batch-size 128\
  --model DiT-S/8\
  --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --image-size 256 \
  --wandb

torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py --model DiT-S/8 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/


###
torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py \
 --model DiT-DEQ-S/8 \
 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ \
 --sig DEQ_simulate_correct1 \
 --wandb \
 --log-every 100

 torchrun --nnodes=1 --nproc_per_node=4 --master_port 28473 train.py \
 --model DiT-DEQ-S/8 \
 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ \
 --sig DEQ_simulate_correct1_with_sample \
 --wandb \
 --sample


 torchrun --nnodes=1 --nproc_per_node=2 --master_port 28473 train.py \
 --model DiT-S/8 \
 --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ \
 --sig DEQ_plain_correct2_with_sample \
 --wandb \
 --sample



## EVAL
torchrun --nnodes=1 --nproc_per_node=4 eval.py \
  --model DiT-B/4 \
  --ckpt /work/xingjian/DiT/results/074-DiT-B-4/checkpoints/0050000.pt

python fid.py \
  --model DiT-B/4 \
  --ckpt /work/xingjian/DiT/results/075-DiT-B-4/checkpoints/0050000.pt



torchrun --nnodes=1 --nproc_per_node=4 eval.py \
  --model DiT-DEQ-B/4 \
  --ckpt /work/xingjian/DiT/results/075-DiT-DEQ-B-4/checkpoints/0150000.pt
python fid.py \
  --model DiT-DEQ-B/4 \
  --ckpt /work/xingjian/DiT/results/075-DiT-DEQ-B-4/checkpoints/0150000.pt


