import torch
from torch import nn

ENC_CLAYERS = 3
DEC_CLAYERS = 3

KERNEL_SIZE = 3
MAXPOOL_KERNEL_SIZE = 2



class Autoencoder(nn.Module):
    def __init__(self,
                 cfg):

        super().__init__()
        # ------------------ ENCODER ------------------ #

        # Convolutional layers (Feature extraction)
        encoder_conv_layers = []
        for i in range(0, len(cfg.ENCODER.CNV_OUT_CHANNELS)):
            in_channels = cfg.IMAGE_CHANNELS
            out_channels = cfg.ENCODER.CNV_OUT_CHANNELS[i]
            if i > 0:
                in_channels = cfg.ENCODER.CNV_OUT_CHANNELS[i - 1]

            convolutional_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=KERNEL_SIZE,
                padding=1,
            )
            maxpool = nn.MaxPool2d(
                kernel_size=MAXPOOL_KERNEL_SIZE  # Could need to be a tuple if images are non-square
            )
            relu = nn.ReLU()
            self.encoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    maxpool,
                    relu
                )
            )

        self.feature_extractor = nn.ModuleList(encoder_conv_layers)

        # Convert to fully connected layers
        fc_in_features = cfg.ENCODER.CNV_OUT_CHANNELS[-1] * 4 * 4  # TODO

        # Fully connected layers (Encoding)
        self.encoder_fc_layer_1 = nn.Linear(
            in_features=fc_in_features,
            out_features=cfg.ENCODER.FC_OUT_FEATURES[0]
        )

        self.encoder_fc_layers = nn.ModuleList(
            [self.encoder_layer_1]
        )
        for i in range(1, len(cfg.ENCODER.FC_OUT_FEATURES)):
            out_features = cfg.ENCODER.FC_OUT_FEATURES[i]
            if i == len(cfg.ENCODER.FC_OUT_FEATURES):
                out_features = cfg.ENCODING_SIZE
            self.encoder_layers.append(
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
            self.decoder_fc_layers.append(
                nn.Linear(
                    in_features=cfg.DECODER.FC_OUT_FEATURES[i - 1],
                    out_features=cfg.DECODER.FC_OUT_FEATURES[i]
                )
            )

        # Deconvolutional layers

        decoder_conv_layers = []
        for i in range(0, DEC_CLAYERS):
            in_channels = cfg.DECODER.FC_OUT_FEATURES[-1]
            out_channels = cfg.DECODER.CNV_OUT_CHANNELS[i]
            if i == DEC_CLAYERS - 1:
                out_channels = cfg.IMAGE_CHANNELS

            convolutional_layer = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=1,
            )
            maxpool = nn.MaxPool2d(
                kernel_size=MAXPOOL_KERNEL_SIZE  # Could need to be a tuple if images are non-square
            )
            relu = nn.ReLU()
            self.encoder_conv_layers.append(
                nn.Sequential(
                    convolutional_layer,
                    maxpool,
                    relu
                )
            )

        self.decoder = nn.ModuleList(decoder_conv_layers)

    def encode(self, image):
        features = self.feature_extractor(image)
        encoding = self.encoder_fc_layers(features)
        return encoding

    def decode(self, encoding):
        fc_result = self.decoder_fc_layers(encoding)
        decoded = self.decoder(fc_result)
        return decoded

    def forward(self, image):
        return self.decode(self.encode(image))
