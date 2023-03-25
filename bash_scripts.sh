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
torchrun --nnodes=1 --nproc_per_node=2 train.py \
  --sample-step 1000 --sample-size 64 \
  --epochs 2000 --model DiT-B/4 \
  --data-path /scratch/shared/beegfs/shared-datasets/ImageNet/ILSVRC12/train/ --image-size 256 \
  --num-classes 1000 \
  --log-every 500 --ckpt-every 2000 


# eval
torchrun --nnodes=1 --nproc_per_node=2 sample_ddp.py \
  --model DiT-XL/2 \
  --num-fid-samples 50000 --cfg-scale 1.0 


# srun
srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=gpu \
 --pty --gres=gpu:4 --constraint=gmem48G --nodelist=gnodel2 /bin/zsh

srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=gpu \
  --mem=50G --pty --gres=gpu:4 --constraint=gmem48G /bin/zsh

srun --ntasks=1 --time=48:00:00 --cpus-per-task=12 --partition=gpu \
  --mem=50G --pty --gres=gpu:1 --constraint=gmem48G /bin/zsh


