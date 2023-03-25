# %%
from cleanfid import fid
import os
import numpy as np


# find all files in fdir1
# files = os.listdir(fdir1)
# def create_subset(folder_suffix, num=100):
#     sub_files = np.random.choice(files, num, replace=False)
#     # create a new folder and put the 100 files in it
#     fdir2 = "/work/xingjian/DiT/samples/" + folder_suffix + "/"
#     os.mkdir(fdir2)
#     for file in sub_files:
#         # print(f"copying {fdir1}{file} to {fdir2}{file}")
#         os.system(f"cp {fdir1}{file} {fdir2}{file}")
#     return fdir2


# %%
# for repeat in range(3):
#     for sample_size in [10, 50, 100, 500, 1000]:
#         if repeat > 0 and sample_size == 1000:
#             continue 
#         fdir2 = create_subset(f"DiT-XL-2-pretrained-size-256-vae-ema-cfg-1.0-seed-0-{sample_size}-{repeat}", num=sample_size)
#         print("created subset of size", sample_size)
#         score = fid.compute_fid(fdir2, dataset_name="imagenet", dataset_res=256, dataset_split="trainval70k")
#         print(f"fid of {sample_size} samples on imagenet1k(train): {score}")

# # %%
# print(f"fid score of samples on imagenet1k(train): {score}")

# %%
fdir1 = "/work/xingjian/DiT/samples/DiT-XL-2-pretrained-size-256-vae-ema-cfg-1.5-seed-0-2023-03-03-08-39/"



score = fid.compute_fid(fdir1, dataset_name="ffhq", dataset_res=256, dataset_split="trainval70k")
print(f"fid score of samples on imagenet1k(train): {score}")