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

        # Convert to fully connected layers
        batch_size = self.batch_size
        self.final_conv_x = cfg.IMAGE_SIZE[0] // 2 ** len(cfg.ENCODER.CNV_OUT_CHANNELS)
        self.final_conv_y = cfg.IMAGE_SIZE[0] // 2 ** len(cfg.ENCODER.CNV_OUT_CHANNELS)
        self.final_channels = cfg.ENCODER.CNV_OUT_CHANNELS[-1]
        self.fc_in_features = batch_size * self.final_conv_x * self.final_conv_y * self.final_channels

        # Fully connected layers (Encoding)
        encoder_fc_layer_1 = nn.Linear(
            in_features=self.fc_in_features // batch_size,
            out_features=cfg.ENCODER.FC_OUT_FEATURES[0]
        )

        self.encoder_fc_layers = nn.ModuleList(
            [encoder_fc_layer_1]
        )
        for i in range(1, len(cfg.ENCODER.FC_OUT_FEATURES)):
            out_features = cfg.ENCODER.FC_OUT_FEATURES[i]
            if i == len(cfg.ENCODER.FC_OUT_FEATURES) - 1:
                out_features = cfg.ENCODING_SIZE
            self.encoder_fc_layers.append(
                nn.Linear(
                    in_features=cfg.ENCODER.FC_OUT_FEATURES[i - 1],
                    out_features=out_features
                )
            )



        # ------------------ DECODER ------------------ #
        self.decoder_fc_layer_1 = nn.Linear(
            in_features=cfg.ENCODING_SIZE,
            out_features=cfg.DECODER.FC_OUT_FEATURES[0]
        )

        self.decoder_fc_layers = nn.ModuleList(
            [self.decoder_fc_layer_1]
        )
        for i in range(1, len(cfg.DECODER.FC_OUT_FEATURES)):
            out_features = cfg.DECODER.FC_OUT_FEATURES[i]
            if i == len(cfg.ENCODER.FC_OUT_FEATURES) - 1:
                out_features = cfg.IMAGE_SIZE[0] * cfg.IMAGE_SIZE[1]
            self.decoder_fc_layers.append(
                nn.Linear(
                    in_features=cfg.DECODER.FC_OUT_FEATURES[i - 1],
                    out_features=out_features
                ),
            )
            self.decoder_fc_layers.append(
                nn.ReLU(),
            )

        # Deconvolutional layers

        decoder_conv_layers = []
        DEC_CLAYERS = len(cfg.DECODER.CNV_OUT_CHANNELS)
        for i in range(0, DEC_CLAYERS):

            in_channels = cfg.DECODER.FC_OUT_FEATURES[-1] # TODO REMOVE
            if i == 0:
                in_channels = cfg.ENCODER.CNV_OUT_CHANNELS[-1]
            else:
                in_channels = cfg.DECODER.CNV_OUT_CHANNELS[i-1]
            out_channels = cfg.DECODER.CNV_OUT_CHANNELS[i]
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
            decoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    nn.Upsample(
                        mode="bilinear",
                        scale_factor=2
                    ),
                    relu
                )
            )

        self.decoder = nn.ModuleList(decoder_conv_layers)

    def encode(self, image):

        encoding_layer_outputs = []
        for layer in self.feature_extractor:
            image = layer(image)
            if True: # TODO configure regularization / feature visualization
                encoding_layer_outputs.append(image)
        """
        image = image.view(image.size(0), self.fc_in_features // image.size(0))
        for layer in self.encoder_fc_layers:
            image = layer(image)"""

        encoding = image
        return encoding, encoding_layer_outputs

    def decode(self, encoding):
        """ for layer in self.decoder_fc_layers:
            encoding = layer(encoding)
        print(encoding.shape)
        encoding = encoding.view(self.batch_size, self.image_channels, self.final_conv_x, self.final_conv_y)
        """
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
