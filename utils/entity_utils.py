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

import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger

"""
def create_frcnn(cfg):
    pascal_classes = np.asarray(['__background__',
                                 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])
    model = resnet(classes=pascal_classes, pretrained=True, class_agnostic=False)
    model.create_architecture()
    model.to(get_device())
    return model
"""
def create_target(cfg):
    model = SSDDetector(cfg)
    model.to(get_device())
    checkpointer = SSDCheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(use_latest=True)
    return model

def create_bb_target(cfg):
    cfg = cfg.clone()  # cfg can be modified by model
    model = build_model(cfg)
    model.eval()
    if len(cfg.DATASETS.TEST):
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model

def create_encoder(cfg):
    optimizers = []
    if cfg.MODEL.MODEL_NAME == "autoencoder":
        model = enc.Autoencoder(cfg)
        checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.load(use_latest=True)
    elif cfg.MODEL.MODEL_NAME == "gan" or cfg.MODEL.MODEL_NAME=="gan_object_detector":
        model = GanEncoder(cfg)
        model.to(get_device())
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
        #checkpointer["discriminator"].load()
        checkpointer["generator"].load()
        checkpointer["encoder"].load(use_latest=True)
        gen_optimizer = torch.optim.Adam(model.encoder_generator.parameters(), lr=0.0002)
        disc_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=0.0002)
        optimizers.append(gen_optimizer)
        optimizers.append(disc_optimizer)
    else:
        raise NotImplementedError("Encoder type not implemented")

    return model, optimizers

