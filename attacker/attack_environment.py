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

import autoencoder.models.autoencoder as enc
from gym.spaces.box import Box
from gym.spaces.tuple import Tuple

from SSD.ssd.data.datasets.evaluation.voc.eval_detection_voc import *



def create_target(cfg):
    model = SSDDetector(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(use_latest=True)
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

    def calculate_map(self, image):
        preds = self.target(image, targets=self.image_data[1]) #TODO device

        prec, rec = calc_detection_voc_prec_rec([preds[0]["boxes"].detach().to("cpu").numpy()],
                                                [preds[0]["labels"].detach().to("cpu").numpy()],
                                                [preds[0]["scores"].detach().to("cpu").numpy()],
                                                [self.annotations[1][0]],
                                                [self.annotations[1][1]],
                                                None,
                                                iou_thresh=0.5)

        with torch.no_grad():
            ap = calc_detection_voc_ap(prec, rec, use_07_metric=False)

        return np.nan_to_num(ap).mean()

    def calculate_reward(self, perturbed_image, original_image):

        map1 = self.calculate_map(perturbed_image)
        map2 = self.calculate_map(original_image)

        performance_reduction_factor = self.attacker_cfg.REWARD.PERFORMANCE_REDUCTION_FACTOR
        delta_factor = self.attacker_cfg.REWARD.DELTA_FACTOR
        reward = performance_reduction_factor * (map1 - map2) - delta_factor * torch.norm(perturbed_image - original_image).detach().cpu().numpy()
        print(map1, map2)
        return reward

    def apply_transformation(self, delta):
        return self.image + delta

    # override
    def step(self, action):
        # get perturbed encoding by applying action
        perturbed_encoding = self.encoding.flatten() + action

        # decode the perturbed encoding to generate a transformation
        perturbation_transformation, _ = self.encoder_decoder.decode(torch.Tensor(perturbed_encoding.reshape(1, 3, 4, 4)), self.encoding_pooling_output)

        # perturb the current image
        perturbed_image = self.apply_transformation(perturbation_transformation)

        # calculate reward based on perturbed image
        reward = self.calculate_reward(self.image, perturbed_image)
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


