from utils.torch_utils import get_device
from autoencoder.models.autoencoder import Autoencoder
from data_management.checkpoint import CheckPointer
from data_management.datasets.image_dataset import ImageDataset
import logging
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import cv2

import os

def save_decod_img(img, epoch):
    img = img.view(img.size(0), 1, 32, 32)
    save_image(img, './MNIST_Out_Images/Autoencoder_image{}.png'.format(epoch))

def make_dir():
    image_dir = 'MNIST_Out_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        checkpointer,
        arguments,
        args
    ):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    make_dir()
    model.train()
    criterion = torch.nn.MSELoss()

    for i in range(0, 200):
        for iteration, (images, _) in enumerate(data_loader):

            iteration += 1
            arguments["iteration"] = iteration

            images = images.to(get_device())
            optimizer.zero_grad()
            reconstructed_images=model(images)

            if iteration % 500 == 0:
                save_decod_img(reconstructed_images.cpu().data, i)

            loss = criterion(images.view(images.size(0), -1), reconstructed_images.view(reconstructed_images.size(0), -1))
            loss.backward()

            optimizer.step()

            print(iteration, loss)

        if i % 5 == 0:
            save_decod_img(reconstructed_images.cpu().data, i)


def start_train(cfg): 
    logger = logging.getLogger('SSD.trainer')

    model = Autoencoder(cfg=cfg)
    model.train()
    model.to(get_device())


    optimizers = {
        "SGD": lambda: torch.optim.SGD(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            momentum=cfg.SOLVER.MOMENTUM,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        ),
        "Adam": lambda: torch.optim.Adam(
            model.parameters(),
            lr=cfg.SOLVER.LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    }

    optimizer = optimizers[cfg.SOLVER.WHICH_OPTIMIZER]()
    transform = transforms.Compose([
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    trainset, testset = None, None

    if True:
        trainset = datasets.FashionMNIST(
            root='./datasetss',
            train=True,
            download=True,
            transform=transform
        )
        testset = datasets.FashionMNIST(
            root='./datasetss',
            train=False,
            download=True,
            transform=transform
        )

    elif False:
        trainset = ImageDataset(transform=transform)
        testset = ImageDataset(transform=transform)

    elif cfg.DATASET_NAME == "coco":
        trainset = datasets.CocoDetection(
            root='../datasets/Coco',
            transform=transform,
            annFile='../datasets/Coco/annotations/instances_train2017.json'
        )
        testset = datasets.CocoDetection(
            root='../datasets/Coco',
            transform=transform,
            annFile='../datasets/Coco/annotations/instances_val2017.json'
        )
    train_loader = DataLoader(
        trainset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    testloader = DataLoader(
        testset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    arguments = {"iteration": 0}
    save_to_disk = True
    checkpointer = CheckPointer(
        model, optimizer, cfg.OUTPUT_DIR, save_to_disk, logger,
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER
    #train_loader = make_data_loader(cfg, is_train=True, max_iter=max_iter, start_iter=arguments['iteration'])

    model = do_train(
        cfg, model, train_loader, optimizer,
        checkpointer, arguments, None)
    return model
    pass