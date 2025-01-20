import os
import torch

def get_time_embedding(timestep, dtype):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=dtype) / 160)
    x = torch.tensor([timestep], dtype=dtype)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def move_channel(image, to):
    if to == "first":
        return image.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    elif to == "last":
        return image.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
    else:
        raise ValueError("to must be one of the following: first, last")

def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_file_path(filename):
    dir_location = os.path.dirname(os.path.abspath(__file__))
    file_location = os.path.join(dir_location, "data", filename)
    return file_location