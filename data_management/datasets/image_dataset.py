from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import os
import io
from skimage import io
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root_dir="C:/Workspace/MasterProject/datasets/Cocotest/train", transform=None):
        super()
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.filenames)
    
    
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i.tolist()

        image_name = os.path.join(self.root_dir,
                                self.filenames[i])
        image = io.imread(image_name)
        if self.transform:
            image = self.transform(image)

        return image
        



