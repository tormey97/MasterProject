import gym
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import abc
import argparse
import copy
import time
from tqdm import tqdm
from SSD.ssd.modeling.detector.ssd_detector import SSDDetector


def create_object_detector(cfg):

    pass


class AttackEnvironment(gym.Env, cfg):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.current_step = 0
        self.obj_detector = None #TODO
        self.autoencoder = None
        self.data_loader = None # Loads the data.

    def calculate_map(self):
        pass

    def calculate_reward(self):

        loss_original = 0
        loss_perturbed = 0
        perturbation_delta = 0

        return loss_reward_factor * (loss_original - loss_perturbed) - delta_reward_factor * perturbation_delta

    def apply_transformation(self, delta):
        self.current_image -= delta

    def take_action(self, action):
        """
        Takes a 10x1 array?
        :param action:
        :return:
        """

    def step(self, action):
        self.take_action(action)
        self.current_step += 1

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
