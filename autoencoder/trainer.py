from utils.torch_utils import get_device
from autoencoder.models.autoencoder import Autoencoder
from autoencoder.models.gan import Network as GANEncoder
from autoencoder.models.test_enc import Autoencoder as at
from autoencoder.models.test_cnv import ConvAutoencoder
from data_management.checkpoint import CheckPointer
from data_management.datasets.image_dataset import ImageDataset
from data_management.datasets.pascal_voc import VocDataset

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

MAX_FILTERS_TO_VISUALIZE = 10
def save_decod_img(img, epoch, cfg, w=None, h=None):
    if w is None or h is None:
        w = cfg.IMAGE_SIZE[0]
        h = cfg.IMAGE_SIZE[1]
    chan = img.size(1)
    if len(img.shape) > 2 and img.size(1) != cfg.IMAGE_CHANNELS:
        chan = [i for i in range(img.size(1))]
    # convolutional filter visualization
    if type(chan) == list:
        visualized = 0
        for r in np.rollaxis(img.detach().cpu().numpy(), 1):
            visualized += 1
            if visualized >= MAX_FILTERS_TO_VISUALIZE:
                break
            if get_device() == "cpu":
                pass
                #save_image(torch.Tensor(r), './autoenc_out/{}Autoencoder_image.png'.format(epoch))
    else:
        img = img.view(img.size(0), chan, w, h)
        save_image(img, './autoenc_out/Autoencoder_image{}.png'.format(epoch))


def make_dir():
    image_dir = 'MNIST_Out_Images'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

def visualize_layer_chain(recon_img, target_img, enc_outputs, dec_outputs, cfg):
    # TODO show activations
    pass

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
    if cfg.SOLVER.LOSS_FUNCTION == "BCE":
        criterion = torch.nn.BCELoss()

    if cfg.MODEL.MODEL_NAME == "gan":
        gen_optim = torch.optim.Adam(model.encoder_generator.parameters(), lr=0.001)
        disc_optim = torch.optim.Adam(model.discriminator.parameters(), lr=0.001)

    iteration = arguments["iteration"]
    for epoch in range(0, 10000):
        last_log = 0

        for i, (images) in enumerate(data_loader):

            iteration += 1
            arguments["iteration"] = iteration

            # Convert to cuda
            images = images.to(get_device())

            # Run batch on model
            if cfg.MODEL.MODEL_NAME == "gan":
                reconstructed_images, encoding = model.encoder_generator(images)
                discriminator_rec = model.discriminator(reconstructed_images)
                discriminator_real = model.discriminator(images)

                disc_loss_real = torch.nn.BCELoss()(discriminator_real, torch.ones_like(discriminator_real))
                disc_loss_rec = 3 * torch.nn.BCELoss()(discriminator_rec, torch.zeros_like(discriminator_rec))

                disc_loss = disc_loss_real + disc_loss_rec

                gen_loss = torch.nn.BCELoss()(discriminator_rec, torch.ones_like(discriminator_rec))

                distortion_penalty = 5 * torch.nn.MSELoss()(reconstructed_images, images)
                gen_loss += distortion_penalty

                gen_optim.zero_grad()
                disc_optim.zero_grad()

                gen_loss.backward(retain_graph=True)
                disc_loss.backward()

                gen_optim.step()
                disc_optim.step()
                print(gen_loss)
                print(disc_loss, disc_loss_real, disc_loss_rec)
                save_decod_img(images.cpu().data, "TARGET" + str(iteration) + "gud", cfg)
                save_decod_img(reconstructed_images.cpu().data, "RECONSTRUCTION" + str(epoch) + "_" + (str(iteration)),
                               cfg)
                #generator loss:

            elif cfg.MODEL.MODEL_NAME == "autoencoder":
                reconstructed_images, dec_outputs, enc_outputs = model(images)

                # -------- L1 Regularization -------- #
                added_loss = 0
                if cfg.SOLVER.L1_REGULARIZATION_FACTOR > 0:
                    l1_enc = sum([torch.norm(output) for output in enc_outputs])
                    #l1_dec = sum([torch.norm(output) for output in dec_outputs])
                    added_loss += cfg.SOLVER.L1_REGULARIZATION_FACTOR * (l1_enc)
                optimizer.zero_grad()

                # Calculate loss
                loss = criterion(reconstructed_images, images) + added_loss,
                loss[0].backward()

                optimizer.step()

                # Print
                if iteration == 0 or iteration - last_log >= cfg.LOG_STEP:
                    last_log = iteration
                    logger.info("ITERATION: ", iteration, "LOSS: ", loss, "ENC_LOSS: ", added_loss)
                    print("_ITERATION: ", iteration, "LOSS: ", loss, "ENC_LOSS: ", added_loss)

                    save_decod_img(images.cpu().data, "TARGET" + str(iteration) + "gud", cfg)
                    save_decod_img(reconstructed_images.cpu().data, "RECONSTRUCTION" + str(epoch)+"_"+(str(iteration)), cfg)
                    # Visualizing output features
                    for i in range(len(enc_outputs)):
                        pass
                        save_decod_img(enc_outputs[i],"ENCODING" + str(epoch) + "_" + str(iteration) + "_" + str(i) + "_" + "enc", cfg, w=enc_outputs[i].shape[2], h=enc_outputs[i].shape[3])
                        save_decod_img(dec_outputs[i],"DECODING" + str(epoch) + "_" +  str(iteration) + "_" + str(i) + "_" + "dec", cfg, w=dec_outputs[i].shape[2],
                                       h=dec_outputs[i].shape[3])

                if iteration % cfg.MODEL.SAVE_STEP == 0:
                    print("SAVING MODEL AT ITERATION ", iteration)
                    checkpointer.save("model_{:06d}".format(iteration), **arguments)



def start_train(cfg): 
    logger = logging.getLogger('SSD.trainer')
    models = {
        "autoencoder": Autoencoder,
        "gan": GANEncoder
    }
    model = models[cfg.MODEL.MODEL_NAME](cfg=cfg)
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
        transforms.ToPILImage(),
        transforms.Resize(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    trainset, testset = None, None

    if cfg.DATASET_NAME == "mnist_fashion":
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
    elif cfg.DATASET_NAME == "test":
        trainset = ImageDataset(transform=transform)
        testset = ImageDataset(transform=transform)
    elif cfg.DATASET_NAME == "voc":
        trainset = VocDataset(
            download=True,
            root='../datasets/Voc',
            transform=transform,
            image_set="train"
        )

        testset = VocDataset(
            download=True,
            root='../datasets/Voc',
            transform=transform,
            image_set="val"
        )
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

    model = do_train(
        cfg, model, train_loader, optimizer,
        checkpointer, arguments, None)
    return model
    pass