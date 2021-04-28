import gym
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import os
import abc
import argparse
import copy
import time
from tqdm import tqdm
from SSD.ssd.modeling.detector.ssd_detector import SSDDetector
from data_management.checkpoint import CheckPointer
from SSD.ssd.utils.checkpoint import CheckPointer as SSDCheckPointer
import autoencoder.models.autoencoder as enc
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple

from SSD.ssd.data.datasets.evaluation.voc.eval_detection_voc import *

import SSD.ssd.data.transforms as detection_transforms
import SSD.ssd.data.transforms.target_transform as target_transforms
from SSD.ssd.data.transforms.transforms import Resize, ToCV2Image, ToTensor

from PIL import Image
from vizer.draw import draw_boxes
from SSD.ssd.data.datasets import COCODataset, VOCDataset


def create_target(cfg):
    model = SSDDetector(cfg)
    checkpointer = SSDCheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load('https://github.com/lufficc/SSD/releases/download/1.2/vgg_ssd300_voc0712.pth', use_latest=False)
    return model


def create_encoder(cfg):
    model = enc.Autoencoder(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(use_latest=True)
    return model


class AttackEnvironment(gym.Env):
    def __init__(self, attacker_cfg, target_cfg, encoder_cfg, data_loader: torch.utils.data.DataLoader):
        super().__init__()
        self.attacker_cfg = attacker_cfg
        self.target_cfg = target_cfg
        self.encoder_cfg = encoder_cfg

        self.target = create_target(target_cfg)
        self.target.eval()
        self.encoder_decoder = create_encoder(encoder_cfg)
        self.data_loader = data_loader

        self.dataset_iterable = enumerate(self.data_loader)

        # Current image and annotations/targets for that image
        self.image_data = None
        self.image = None
        self.annotations = None
        # Encoding of current image
        self.encoding = None
        self.encoding_pooling_output = None

        self.action_space = Box(-1, 1, [48])  # TODO configurable
        self.observation_space = Box(-1, 1, [48])

        self.step_ctr = 0

    def calculate_map(self, image):
        transform = detection_transforms.Compose([
            detection_transforms.ToCV2Image(),
            detection_transforms.ConvertFromInts(),
            detection_transforms.ToPercentCoords(),
            detection_transforms.Resize(300),
            ToTensor(),
        ])

        transform2 = detection_transforms.Compose([
            detection_transforms.ToCV2Image(),
            detection_transforms.ConvertFromInts(),
            detection_transforms.ToPercentCoords(),
            detection_transforms.Resize(160),
            detection_transforms.ToAbsoluteCoords(),
            ToTensor(),
        ])

        image = image[0]
        boxes = self.image_data[1]["boxes"][0]
        labels = self.image_data[1]["labels"][0]
        _, gt_boxes, gt_labels = transform2(image, self.annotations[1][0], self.annotations[1][1])

        image, boxes, labels = transform(image, boxes, labels)

        targets = {'boxes': boxes, 'labels': labels}
        preds = self.target(image.unsqueeze(0), targets=targets)[0] #TODO device


        pred_boxes = preds["boxes"].detach().to("cpu").numpy()
        pred_labels = preds["labels"].detach().to("cpu").numpy()
        pred_scores = preds["scores"].detach().to("cpu").numpy()
        indices = pred_scores > 0.9
        pred_boxes = pred_boxes[indices]
        pred_labels = pred_labels[indices]
        pred_scores = pred_scores[indices]
        cv2image = image.detach().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

        drawn_image = draw_boxes(cv2image, pred_boxes, pred_labels, pred_scores, VOCDataset.class_names).astype(np.uint8)
        Image.fromarray(drawn_image).save(os.path.join("justtosee", str(self.step_ctr) + "asdfdg.jpg"))
        self.step_ctr += 1
        prec, rec = calc_detection_voc_prec_rec([pred_boxes],
                                                [pred_labels],
                                                [pred_scores],
                                                [gt_boxes],
                                                [gt_labels],
                                                None,
                                                iou_thresh=0.5)

        with torch.no_grad():
            ap = calc_detection_voc_ap(prec, rec, use_07_metric=False)

        return np.nan_to_num(ap).mean()

    def calculate_reward(self, perturbed_image, original_image, perturbation):

        map_perturbed = self.calculate_map(perturbed_image)
        map_orig = self.calculate_map(original_image.detach())

        performance_reduction_factor = self.attacker_cfg.REWARD.PERFORMANCE_REDUCTION_FACTOR
        delta_factor = self.attacker_cfg.REWARD.DELTA_FACTOR
        diff = np.linalg.norm(perturbation)
        reward = performance_reduction_factor * (map_orig - map_perturbed) - delta_factor * diff
        print(map_perturbed, map_orig)
        print(reward)
        return reward

    def apply_transformation(self, delta):
        perturbed_image = (self.image + delta * 255) / 2
        return perturbed_image

    # override
    def step(self, action):
        # get perturbed encoding by applying action
        perturbed_encoding = self.encoding.flatten() + action

        # decode the perturbed encoding to generate a transformation
        reconstruction, _ = self.encoder_decoder.decode(torch.Tensor(self.encoding), self.encoding_pooling_output)
        perturbation_transformation, _ = self.encoder_decoder.decode(torch.Tensor(perturbed_encoding.reshape(1, 3, 4, 4)), self.encoding_pooling_output)
        perturbation_transformation = perturbation_transformation - reconstruction
        # perturb the current image
        perturbed_image = self.apply_transformation(perturbation_transformation)

        # calculate reward based on perturbed image
        reward = self.calculate_reward(self.image, perturbed_image, perturbation_transformation.detach().cpu().numpy())
        done = True  # Done is always true, we consider one episode as one image
        return perturbed_encoding.flatten(), reward, done, {}

    #override
    def reset(self):
        """
        :return: the initial state of the problem, which is an encoding of the image
        """
        i = 0
        try:
            i, values = next(self.dataset_iterable)

        except StopIteration:
            self.dataset_iterable = enumerate(self.data_loader)
            i, values = next(self.dataset_iterable)
        self.annotations = self.data_loader.dataset.get_annotation(i)
        self.image_data = values
        self.image = values[0]
        self.encoding = self.encoder_decoder.encode(self.image)[0].detach().cpu().numpy()
        return self.encoding.flatten()


    #override
    def render(self, mode='human'):
        pass

    def close(self):
        pass

def create_env(attacker_config, encoder_config, target_config, data_loader):
    return AttackEnvironment(target_cfg=target_config, data_loader=data_loader, attacker_cfg=attacker_config, encoder_cfg=encoder_config)


