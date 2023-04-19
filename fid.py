
 

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
import plotly.graph_objs as go
import plotly.io as pio


from generate import calc_fid
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=1024)

    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--num-sampling-iterations", type=int, default=12)

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.0)
    parser.add_argument("--global-seed", type=int, default=42)

    parser.add_argument("--ckpt", type=str,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()

    from eval import num_sampling_steps, num_sampling_iterations

    wandb.init(project="DiT-self-trained", config=args, name="FID_sweep" + args.model)

    # Initialize the heatmap data
    heatmap_data = np.zeros((len(num_sampling_steps), len(num_sampling_iterations)))

    for i, num_sampling_step in enumerate(num_sampling_steps):
        for j, num_sampling_iteration in enumerate(num_sampling_iterations):
            train_args = {"num_sampling_steps": num_sampling_step, "num_sampling_iterations": num_sampling_iteration}

            score = calc_fid(args, train_args)

            # Update the heatmap data
            heatmap_data[i, j] = score
            print(f"calculated FID score {score} for {train_args}")

    # Create discrete heatmap
    # add labels and captions
    discrete_heatmap = go.Figure(go.Heatmap(
        x=num_sampling_iterations,
        y=num_sampling_steps,
        z=heatmap_data,
        colorscale="Viridis",
        showscale=True,
    ))

    # Create continuous heatmap
    continuous_heatmap = go.Figure(go.Heatmap(
        x=num_sampling_iterations,
        y=num_sampling_steps,
        z=heatmap_data,
        colorscale="Viridis",
        showscale=True,
        zsmooth="best",
    ))

    # Save the heatmaps locally
    # random number as suffix
    rand = np.random.randint(1000)
    pio.write_image(discrete_heatmap, f"discrete_heatmap-{str(rand)}.png")
    pio.write_image(continuous_heatmap, f"continuous_heatmap-{str(rand)}.png")

    wandb.log({"discrete_heatmap": wandb.Image(f"discrete_heatmap-{str(rand)}.png")})
    wandb.log({"continuous_heatmap": wandb.Image(f"continuous_heatmap-{str(rand)}.png")})

   
    # sweep_config = {
    #     'method': 'grid',
    #     'name': 'hyperparameter_sweep',
    #     'metric': {
    #         'goal': 'minimize',
    #         'name': 'FID_score'
    #     },
    #     'parameters': {
    #         'num_sampling_steps': {
    #             'values': num_sampling_steps
    #         },
    #         'num_sampling_iterations': {
    #             'values': num_sampling_iterations
    #         }
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_config, project="DiT-self-trained")
    
    # def calc_fid_sweep():
    #     #assert there is only one process
    #     # assert dist.get_world_size() == 1
    #     # Parse command-line arguments
    #     wandb.init()
    #     train_args = wandb.config
    #     print(f"calc fid sweep with {train_args}")
    #     score = calc_fid(args, train_args)
    #     wandb.log({"FID_score": score})
    #     wandb.log({"num_sampling_steps": train_args.num_sampling_steps})
    #     wandb.log({"num_sampling_iterations": train_args.num_sampling_iterations})

    # wandb.agent(sweep_id, calc_fid_sweep)