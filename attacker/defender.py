import torch
from utils.torch_utils import get_device
def defender(images, noise_levels):
    new_images = []
    for lvl in noise_levels:
        if lvl > 0:
            noise = torch.normal(0, lvl, size=images.shape).to(get_device())
            new_image = torch.add(images, noise)
            new_images.append(new_image)
        else:
            new_images.append(images)
    return new_images

