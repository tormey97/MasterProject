# here we will create datasets (e.g. apply adversarial attacks and add them to datasets, create datasets with probability from detecotr model etc)

from SSD.ssd.data.datasets.mnist import MNISTDetection
from SSD.ssd.data.build import make_data_loader

class MNIST_Dataloader():
    def __init__(self, cfg):
        self.dataloader = make_data_loader(cfg, is_train=True, distributed=False, max_iter=None, start_iter=0)