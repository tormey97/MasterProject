from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import VOCDetection
import torch
import os
import io
from skimage import io
import cv2
import numpy as np


class VocDataset(VOCDetection):
    def __init__(self, root="~/data/pascal_voc", image_set="train", download=True, transform=None):
        super().__init__(root=root, image_set=image_set, download=True, transform=transform)
        self.transform = transform

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i.tolist()
        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image




