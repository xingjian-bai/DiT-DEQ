# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

import wandb
from cleanfid import fid


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def generate(args, num_sampling_steps = None, num_sampling_iterations = None, tf32 = True, rank = 0, device = 0):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = tf32  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)

   
    if num_sampling_steps is None:
        num_sampling_steps = args.num_sampling_steps
    if num_sampling_iterations is None:
        num_sampling_iterations = args.num_sampling_iterations


    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    
    model.load_state_dict(state_dict)
    model.eval()  # important!
    # change the sample step of DEQ inside the model to num_sampling_iterations
    if model.deq_mode == 'simulate_repeat':
        model.deq.eval_f_thres = num_sampling_iterations
    elif model.deq_mode == None:
        pass
    else:
        raise NotImplementedError
    
    diffusion = create_diffusion(str(num_sampling_steps))
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}" \
                  f"-steps-{num_sampling_steps}-iters-{num_sampling_iterations}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    # if it exists, we delete all the files in the folder
    if os.path.exists(sample_folder_dir):
        files = os.listdir(sample_folder_dir)
        for file in files:
            if rank == 0:
                os.remove(os.path.join(sample_folder_dir, file))

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    
    # if rank == 0:
    # print(f"model: {args.model}, num_fid_samples: {args.num_fid_samples}, num_sampling_steps: {args.num_sampling_steps}, ckpt: {args.ckpt}")
        # print(f"fid score: {score}")
    
    return sample_folder_dir

#import namespace
from argparse import Namespace
def calc_fid (args, train_args): #WITH ONLY ONE GPU
    # if train_args is a dict, we convert it to a Namespace object
    if isinstance(train_args, dict):
        train_args = Namespace(**train_args)

    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}" \
                  f"-steps-{train_args.num_sampling_steps}-iters-{train_args.num_sampling_iterations}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    print(f"FID for {sample_folder_dir}")
    score = fid.compute_fid(sample_folder_dir, dataset_name="imagenet", dataset_res=256)

    files = os.listdir(sample_folder_dir)
    for file in files[:max(0, -16)]:
        os.remove(os.path.join(sample_folder_dir, file))
    return score