import os
import abc
import torch
import argparse
import copy
import time
from tqdm import tqdm
from SSD.ssd.modeling.detector.ssd_detector import SSDDetector
from data_management.checkpoint import CheckPointer
from SSD.ssd.utils.checkpoint import CheckPointer as SSDCheckPointer
import autoencoder.models.autoencoder as enc
from autoencoder.models.gan import Network as GanEncoder
#from autoencoder.trainer import save_decod_img
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple

#from autoencoder.trainer import save_decod_img

from SSD.ssd.data.datasets.evaluation.voc.eval_detection_voc import *

import SSD.ssd.data.transforms as detection_transforms
import SSD.ssd.data.transforms.target_transform as target_transforms
from SSD.ssd.data.transforms.transforms import Resize, ToCV2Image, ToTensor

from PIL import Image
from vizer.draw import draw_boxes
from SSD.ssd.data.datasets import COCODataset, VOCDataset

from utils.torch_utils import get_device




def create_target(cfg):
    model = SSDDetector(cfg)
    model.to(get_device())
    checkpointer = SSDCheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load('https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth', use_latest=False)
    return model


def create_encoder(cfg):
    optimizers = []
    if cfg.MODEL.MODEL_NAME == "autoencoder":
        model = enc.Autoencoder(cfg)
        checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(use_latest=True)
    elif cfg.MODEL.MODEL_NAME == "gan" or cfg.MODEL.MODEL_NAME=="gan_object_detector":
        model = GanEncoder(cfg)
        checkpointer = {
            "discriminator": CheckPointer(
                model.discriminator, save_dir=cfg.OUTPUT_DIR,
                last_checkpoint_name="disc_chkpt.txt"
            ),
            "generator": CheckPointer(
                model.encoder_generator, save_dir=cfg.OUTPUT_DIR,
                last_checkpoint_name="gen_chkpt.txt"
            ),
            "encoder": CheckPointer(
                model.encoder_generator.encoder, save_dir=cfg.OUTPUT_DIR, last_checkpoint_name="encoder_chkpt.txt"
            )
        }
        checkpointer["discriminator"].load()
        #checkpointer["generator"].load()
        checkpointer["encoder"].load(use_latest=True)
        gen_optimizer = torch.optim.Adam(model.encoder_generator.parameters(), lr=0.0002)
        disc_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002)
        optimizers.append(gen_optimizer)
        optimizers.append(disc_optimizer)
    else:
        raise NotImplementedError("Encoder type not implemented")

    return model, optimizers