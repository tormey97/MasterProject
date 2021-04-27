import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np


KERNEL_SIZE = 5
MAXPOOL_KERNEL_SIZE = 2



class Autoencoder(nn.Module):
    def __init__(self,
                 cfg):

        super().__init__()
        # ------------------ ENCODER ------------------ #

        # Convolutional layers (Feature extraction)
        encoder_conv_layers = []
        encoder_max_pools = []
        self.cfg = cfg
        self.image_channels = cfg.IMAGE_CHANNELS
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        for i in range(0, len(cfg.ENCODER.CNV_OUT_CHANNELS)):
            in_channels = cfg.IMAGE_CHANNELS
            out_channels = cfg.ENCODER.CNV_OUT_CHANNELS[i]
            if i > 0:
                in_channels = cfg.ENCODER.CNV_OUT_CHANNELS[i - 1]

            convolutional_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=KERNEL_SIZE,
                padding=2,
                stride=2,
            )
            maxpool = nn.MaxPool2d(
                kernel_size=MAXPOOL_KERNEL_SIZE,  # Could need to be a tuple if images are non-square
                return_indices=True
            )
            encoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(p=0.05),
                )
            )
            encoder_max_pools.append(
                maxpool
            )

        self.feature_extractor = nn.ModuleList(encoder_conv_layers)
        self.encoder_maxpools = nn.ModuleList(encoder_max_pools)

        # ------------------ DECODER ------------------
        decoder_conv_layers = []
        decoder_unpooling = []

        DEC_CLAYERS = len(cfg.DECODER.CNV_OUT_CHANNELS)
        for i in range(0, DEC_CLAYERS):
            # First layer takes in encoding
            in_channels = cfg.ENCODER.CNV_OUT_CHANNELS[-1]
            if i != 0:
                in_channels = cfg.DECODER.CNV_OUT_CHANNELS[i - 1]

            out_channels = cfg.DECODER.CNV_OUT_CHANNELS[i]
            # Last layer should return an RGB or greyscale image
            if i == DEC_CLAYERS - 1:
                out_channels = cfg.IMAGE_CHANNELS

            convolutional_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            )

            if i == DEC_CLAYERS - 1:
                torch.nn.init.uniform_(convolutional_layer.weight, -2, 2)
            else:
                torch.nn.init.xavier_uniform_(convolutional_layer.weight)

            relu = nn.ReLU()
            if i == DEC_CLAYERS - 1:
                if cfg.SOLVER.LOSS_FUNCTION == "BCE":
                    relu = nn.Sigmoid()
                else:
                    relu = nn.Sigmoid()

            unpool = nn.MaxUnpool2d(
                        kernel_size=MAXPOOL_KERNEL_SIZE,
                    )

            decoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    nn.Upsample(
                        scale_factor=2,
                        mode="bilinear"
                    ),
                    relu,
                )
            )
            decoder_unpooling.append(
                unpool
            )

        self.decoder = nn.ModuleList(decoder_conv_layers)
        self.decoder_unpooling = nn.ModuleList(decoder_unpooling)

    def encode(self, image):

        encoding_layer_outputs = []
        pooling_indices = []
        for i, layer in enumerate(self.feature_extractor):
            image = layer(image)
            if True: # TODO configure regularization / feature visualization
                encoding_layer_outputs.append(image)
            image, indices = self.encoder_maxpools[i](image)
           # pooling_indices.append(indices)
        for i in range(self.cfg.MODEL.AVG_POOL_COUNT):
            image = F.avg_pool2d(image, 2)
        encoding = image

        pooling_indices.reverse()
        return encoding, encoding_layer_outputs, pooling_indices

    def decode(self, encoding, pooling_indices):
        decoding_layer_outputs = []
        for i in range(self.cfg.MODEL.AVG_POOL_COUNT):
            encoding = F.upsample_bilinear(encoding, scale_factor=2)
        for i, layer in enumerate(self.decoder):
            # encoding = self.decoder_unpooling[i](encoding, pooling_indices[i])
            encoding = layer(encoding)
            if True: # TODO configure regularization / feature visualization
                decoding_layer_outputs.append(encoding)
        decoded = encoding
        return decoded, decoding_layer_outputs

    def forward(self, image):
        encoding, encoding_layer_outputs, pooling_indices = self.encode(image)
        # return the decoding as well as decoding_layer_outputs and encoding_layer_outputs for regularization purposes
        decoded, decoding_layer_outputs = self.decode(encoding, pooling_indices)
        return decoded, decoding_layer_outputs, encoding_layer_outputs
