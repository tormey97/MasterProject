import torch
from torch import nn

# need:
# generator - reconstructs a quantized image that should fool the discrimintaor
# discriminator - takes reconstructed image and gives probability it belongs to original distribution
