import torch
from torch import nn
from torchvision import models

def extra_layer(in_channels, out_channels, num_filters, kernel_size=5):
    STRIDE = 1
    layer = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            stride=STRIDE,
            padding=0,
        ),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=1
        ),
        nn.ReLU(),

        nn.Conv2d(
            in_channels=num_filters,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=STRIDE,
            padding=0,
        ),
        nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=1
        ),
        nn.ReLU(),
    )
    return layer

class ResnetModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        cnv_layers = nn.Sequential(*list(self.resnet.children())[:-4])

        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        self.feature_maps = nn.ModuleList()
        self.model = cnv_layers
        self.filters_per_size = [None, 128, 256, 128, 128, 128]
        num_filters = 64

        for i in range(3, 6):
            conv_stride = 1
            conv_padding = 0
            if i < 6 - 1:
                conv_stride = 2
                conv_padding = 1

            layer = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels= output_channels[i-1],
                    out_channels= self.filters_per_size[i],
                    kernel_size=3,
                    stride=1,
                    padding=1
                  ),
                nn.BatchNorm2d(self.filters_per_size[i]),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=self.filters_per_size[i],
                    out_channels=2 * self.filters_per_size[i],
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2d(2 * self.filters_per_size[i]),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=2 * self.filters_per_size[i],
                    out_channels=self.output_channels[i],
                    kernel_size=3,
                    stride=conv_stride,
                    padding=conv_padding
                )
            )
            self.feature_maps.append(layer)


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = [self.model(x)]
        out_features.append(self.resnet.layer3(out_features[0]))
        out_features.append(self.resnet.layer4(out_features[1]))
        for i in range(0, len(self.feature_maps)):
            output = self.feature_maps[i](out_features[3 + i - 1])
            out_features.append(output)
        return tuple(out_features)

