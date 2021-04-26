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
                kernel_size=MAXPOOL_KERNEL_SIZE  # Could need to be a tuple if images are non-square
            )
            encoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(p=0.05),
                    maxpool,
                )
            )

        self.feature_extractor = nn.ModuleList(encoder_conv_layers)
        self.feature_extractor.append(nn.AvgPool2d(
            kernel_size=MAXPOOL_KERNEL_SIZE,

        ))
        # ------------------ DECODER ------------------
        decoder_conv_layers = []
        decoder_conv_layers.append(nn.Upsample(
            mode="bilinear",
            scale_factor=2
        ))
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
            torch.nn.init.xavier_uniform(convolutional_layer.weight)
            relu = nn.ReLU()
            if i == DEC_CLAYERS - 1:
                relu = nn.Sigmoid()
            upsample = nn.Upsample(
                        mode="bilinear",
                        scale_factor=2
                    )

            decoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    upsample,
                    relu,
                )
            )

        self.decoder = nn.ModuleList(decoder_conv_layers)


    def encode(self, image):

        encoding_layer_outputs = []
        for layer in self.feature_extractor:
            image = layer(image)
            if True: # TODO configure regularization / feature visualization
                encoding_layer_outputs.append(image)
        encoding = image
        return encoding, encoding_layer_outputs

    def decode(self, encoding):
        decoding_layer_outputs = []
        for layer in self.decoder:
            encoding = layer(encoding)
            if True: # TODO configure regularization / feature visualization
                decoding_layer_outputs.append(encoding)

        decoded = encoding
        return decoded, decoding_layer_outputs

    def forward(self, image):
        encoding, encoding_layer_outputs = self.encode(image)
        # return the decoding as well as decoding_layer_outputs and encoding_layer_outputs for regularization purposes
        decoded, decoding_layer_outputs = self.decode(encoding)
        return decoded, decoding_layer_outputs, encoding_layer_outputs
