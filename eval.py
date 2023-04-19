
 

import torch
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")

import wandb
from PIL import Image
import os
from cleanfid import fid

import torch.distributed as dist
from download import find_model
from tqdm import tqdm
import os
import numpy as np
import math
import argparse
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

# %%

def save_images(images, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    for i, image in enumerate(images):
        save_image(image, os.path.join(folder_name, f"{i}.png"))

def compute_fid (sample_folder, dataset_folder):
    score = fid.compute_fid(dataset_folder, sample_folder)
    return score

def evaluation (model, args, sample_size = 8, sample_step = 250):
    model.eval()

    image_size = args.image_size
    latent_size = int(image_size) // 8
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)

    #  Set user inputs:
    seed = 0 #@param {type:"number"}
    noise_generator = torch.Generator(device=device)
    noise_generator.manual_seed(seed)
    # torch.manual_seed(seed) # no!! this is plugged in the training loop

    num_sampling_steps = sample_step #@param {type:"slider", min:0, max:1000, step:1}
    cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
    class_labels = [0] * sample_size #@param {type:"raw"}
    samples_per_row = 4 #@param {type:"number"}

    # Create diffusion object:
    diffusion = create_diffusion(str(num_sampling_steps))

    # Create sampling noise:
    # n = len(class_labels)
    z = torch.randn(sample_size, 4, latent_size, latent_size, device=device, generator=noise_generator)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([0] * sample_size, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, 
        model_kwargs=model_kwargs, progress=False, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    # current time in a string without space:
    return samples

def evaluation_large (model, args, sample_size = 512, sample_step = 250):
    samples = []

    if sample_size <= 8:
        return evaluation(model, args, sample_size, sample_step)

    batch_size = 8
    assert sample_size % batch_size == 0
    for _ in range (0, sample_size, batch_size):
        samples.append(evaluation(model, args, batch_size, sample_step))
    samples = torch.cat(samples, dim=0)
    # print("shape: ", samples.shape)
    return samples

# def evaluate_fid (model, args, sample_size = 8, sample_step = 250):
#     samples = evaluation_large(model, args, sample_size, sample_step)
#     save_images(samples, "sample")
#     score = compute_fid("sample", args.dataset)
#     return score

    
num_sampling_steps = [250]
num_sampling_iterations = [i for i in range (1, 60)]

from generate import generate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=128)
    parser.add_argument("--num-fid-samples", type=int, default=2048)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-sampling-iterations", type=int, default=12)

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--global-seed", type=int, default=42)

    parser.add_argument("--ckpt", type=str,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()




     # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    for num_sampling_step in num_sampling_steps:
        for num_sampling_iteration in num_sampling_iterations:
            generate(args, num_sampling_step, num_sampling_iteration, rank=rank, device=device)
    
    dist.destroy_process_group()