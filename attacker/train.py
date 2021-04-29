import torch
from torch import nn
import numpy as np
import argparse
from autoencoder.configs.defaults import cfg as autoencoder_cfg
from attacker.configs.defaults import cfg as attacker_cfg
from attacker.attack_environment import create_env
import torchvision.transforms as transforms

from SSD.ssd.utils.checkpoint import CheckPointer
from SSD.ssd.utils.logger import setup_logger

from SSD.ssd.config.defaults import _C as target_cfg
import pathlib
from data_management.logger import setup_logger

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

from data_management.datasets.image_dataset import ImageDataset
from data_management.datasets.voc_detection import VOCDataset
from torch.utils.data import DataLoader

from SSD.ssd.data.transforms.transforms import *
from SSD.ssd.data.transforms import build_target_transform

import os

def get_parser():
    parser = argparse.ArgumentParser(description='Attacker')
    parser.add_argument(
        "attacker_config_file",
        default="",
        metavar="FILE",
        help="path to config file for RL attacker",
        type=str,
    )
    parser.add_argument(
        "encoder_config_file",
        default="",
        metavar="FILE",
        help="path to config file for autoencoder",
    )
    parser.add_argument(
        "target_config_file",
        default="",
        metavar="FILE",
        help="path to config file for object detector",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

def start_train(attacker_cfg, encoder_cfg, target_cfg):

    transform = Compose([
        Resize(target_cfg.INPUT.IMAGE_SIZE),
        ToTensor(),
    ])

    target_transform = build_target_transform(target_cfg)

    # trainset = ImageDataset(transform=Transform)
    trainset = VOCDataset(
        data_dir='./datasets/Voc/VOCdevkit/VOC2012',
        transform=transform,
        split="train",
        target_transform=target_transform,
        keep_difficult=True
    )

    data_loader = DataLoader(trainset)

    env = create_env(
        attacker_config=attacker_cfg,
        encoder_config=encoder_cfg,
        target_config=target_cfg,
        data_loader=data_loader
    )

    # the noise objects for DDPG
    n_actions = np.prod(env.action_space.shape)
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape), sigma=float(0.5) * np.ones(env.action_space.shape))
    folder = attacker_cfg.OUTPUT_DIR

    if os.path.exists(folder + "/" + attacker_cfg.OUTPUT_FILE):
        attacker = DDPG.load(folder + "/" + attacker_cfg.OUTPUT_FILE)
        attacker.set_env(env)
    else:
        attacker = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, tensorboard_log="./logs/progress_tensorboard/")
    for i in range(attacker_cfg.TRAIN.SAVE_AMOUNT):
        attacker.learn(attacker_cfg.TRAIN.SAVE_STEP)
        attacker.save(folder + "/" + attacker_cfg.OUTPUT_FILE)



    return attacker


def main():
    args = get_parser().parse_args()
    attacker_cfg.merge_from_file(args.attacker_config_file)
    attacker_cfg.freeze()

    autoencoder_cfg.merge_from_file(args.encoder_config_file)
    autoencoder_cfg.freeze()

    target_cfg.merge_from_file(args.target_config_file)
    target_cfg.freeze()
    output_dir = pathlib.Path(attacker_cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger = setup_logger("Attacker", output_dir)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.attacker_config_file))
    with open(args.attacker_config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(attacker_cfg))

    model = start_train(attacker_cfg, autoencoder_cfg, target_cfg)

    logger.info('Start evaluating...')
    torch.cuda.empty_cache()  # speed up evaluating after training finished





if __name__ == '__main__':
    main()

