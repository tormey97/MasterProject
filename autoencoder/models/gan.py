import torch
from torch import nn
import numpy as np
from utils.torch_utils import get_device
import torch.nn.functional as F
# need:
# encoder - encodes an image down to some size.
# generator - reconstructs a image that should fool the discrimintaor
# discriminator - takes reconstructed image and gives probability it belongs to original distribution

def conv_block(in_channels, out_channels, kernel_size, stride, padding, activ=nn.ReLU):
    cnv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    torch.nn.init.xavier_uniform_(cnv.weight)
    return nn.Sequential(
        cnv,
        nn.InstanceNorm2d(num_features=out_channels),
        activ()
    )

class EncoderGenerator(nn.Module):
    def __init__(self, cfg, f, C, in_channels):
        super().__init__()
        self.f = f
        self.C = C
        self.in_channels = in_channels
        self.generator = self.make_generator()
        self.encoder = self.make_encoder()


    def forward(self, x):
        encoding = self.encoder(x)
        quantized = self.quantizer(encoding)
        recon = self.generator(quantized)
        return recon, encoding, quantized

    def encode(self, x):
        return self.encoder(x)

    def reconstruct_from_encoding(self, encoding):
        """
        attacker modifies unquantized encoding
        :param encoding:
        :return:
        """
        quantized = self.quantizer(encoding)
        recon = self.generator(quantized)
        return recon, quantized,

    def reconstruct_from_quantized_encoding(self, quantized):
        """
        attacker modifies the quantized encoding
        :param quantized:
        :return:
        """
        recon = self.generator(quantized)
        return recon

    def make_encoder(self):
        block1 = conv_block(self.in_channels, self.f[0], 7, 1, 3)
        block2 = conv_block(self.f[0], self.f[1], 3, 2, 1)
        block3 = conv_block(self.f[1], self.f[2], 3, 2, 1)
        block4 = conv_block(self.f[2], self.f[3], 3, 2, 1)
        block5 = conv_block(self.f[3], self.f[4], 3, 2, 1)
        block6 = conv_block(self.f[4], self.f[5], 3, 1, 1)
        return nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,
            block6
        )

    def quantizer(self, x: torch.Tensor):
        # quantizer discretizes the encoding to (-2, -1, 0, 1, 2)
        centers = np.array([-2, -1, 0, 1, 2])
        # create a vector for each value with the distance to centers, then take the index of centers with the smallest
        def to_center(z_i):
            diffs = torch.abs(z_i - torch.Tensor(centers))
            wtf = torch.argmin(
                diffs
            )
            return centers[wtf]

        def to_soft(z_i):
            tensor_centers = torch.Tensor(centers)
            zhat_i = sum([c_j * torch.exp(-1 * torch.abs(z_i - c_j)) / sum([torch.exp(-1 * torch.abs(z_i - c_l))for c_l in tensor_centers] )
                      for c_j in tensor_centers])
            return zhat_i

        x_copy = x.clone().detach().cpu()
        w_hard = x_copy.clone().apply_(
            lambda x: to_center(x)
        )

        w_soft_ = x_copy.clone().apply_(
            lambda z_i: z_i - to_soft(z_i)
        )
        if get_device() == "cuda":
            w_soft_ = w_soft_.cuda()
            w_hard = w_hard.cuda()

        w_return = (w_hard - (x + w_soft_)).detach() + (x + w_soft_)
        return w_return


    def make_generator(self):
        def residual_block(num_filters, stride, kernel_size, actv):
            p = int((kernel_size - 1) / 2)
            conv1 = nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                stride=stride,
                kernel_size=kernel_size,
                padding=p
            )
            norm = nn.InstanceNorm2d(num_filters)
            conv2 = nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                stride=stride,
                kernel_size=kernel_size,
                padding=p
            )
            return nn.Sequential(
                conv1,
                norm,
                actv(),
                conv2,
                norm,
                actv()
            )

        def upsample_block(in_channels, out_channels, stride, kernel_size, padding, actv):
            transpose = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                padding=padding
            )
            nn.init.xavier_uniform_(transpose.weight, 1)
            return nn.Sequential(
                transpose,
                actv()
            )

        ups1 = nn.Conv2d(
            in_channels=self.f[-1],
            out_channels=self.f[-2],
            kernel_size=3,
            stride=1,
            padding=1
        )

        residual_blocks = [residual_block(self.f[-2], 1, 3, nn.ReLU) for _ in range(9)]

        ups2 = upsample_block(self.f[-2], self.f[-3], 2, 2, 0, nn.ReLU)
        ups3 = upsample_block(self.f[-3], self.f[-4], 2, 2, 0, nn.ReLU)
        ups4 = upsample_block(self.f[-4], self.f[-5], 2, 2, 0, nn.ReLU)
        ups5 = upsample_block(self.f[-5], self.f[-6], 2, 2, 0, nn.ReLU)

        to_img = nn.Conv2d(
            in_channels=self.f[-6],
            out_channels=self.in_channels,
            kernel_size=7,
            stride=1,
            padding=3
        )

        actv_out = nn.Tanh()

        return nn.Sequential(
            ups1,
            *residual_blocks,
            ups2,
            ups3,
            ups4,
            ups5,
            to_img,
            actv_out
        )

class Discriminator(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.discriminator = self.make_discriminator()

    def make_discriminator(self, actv=nn.LeakyReLU):
        d_f = [64, 128, 256, 512, 1]
        c1 = conv_block(in_channels=self.in_channels, out_channels=d_f[0], kernel_size=7, stride=1, padding=1, activ=actv)
        c2 = conv_block(in_channels=d_f[0], out_channels=d_f[1], stride=2, padding=0, kernel_size=4, activ=actv)
        c3 = conv_block(in_channels=d_f[1], out_channels=d_f[2], stride=2, padding=0, kernel_size=4, activ=actv)
        c4 = conv_block(in_channels=d_f[2], out_channels=d_f[3], stride=2, padding=0, kernel_size=4, activ=actv)

        out = nn.Conv2d(in_channels=d_f[3], out_channels=d_f[4], kernel_size=4, stride=1, padding=0)
        actv = nn.Sigmoid()
        return nn.Sequential(c1, c2, c3, c4, out, actv)

    def forward(self, x):
        return self.discriminator(x)

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.in_channels = 3
        self.C = 3
        self.f = [60, 120, 240, 480, 960, self.C]
        self.encoder_generator = EncoderGenerator(cfg, self.f, self.C, self.in_channels)
        self.discriminator = Discriminator(cfg, self.in_channels)

    def encgen_forward(self, x):
        return self.encoder_generator(x)


    def gen_forward(self, x):
        return self.generator(x)

    def disc_forward(self, x):
        return self.discriminator(x)

    def encode(self, x):
        return self.encoder_generator.encode(x)

    def decode(self, encoding):
        recon, quantized = self.encoder_generator.reconstruct_from_encoding(encoding)
        return recon, quantized

    def forward(self, x):
        generated, enc, quantized_enc = self.encoder_generator.forward(x)
        return generated, self.discriminator(generated), enc, quantized_enc
