#!/bin/bash

#SBATCH --time=5:00:00
#SBATCH --partition=ddp-2way
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
# SBATCH --constraint=gmem24G


gpustat
date
echo “Job started.”
time python imagenet_eval.py
echo “Job completed.”

