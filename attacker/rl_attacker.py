import stable_baselines3 as stable_baselines
import cv2
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
import gym

'''
* Use PPO?
* Laplacian operator on image to detect important pixels
* Saliency map on image
* "Autoencoder" type attack, where image is reduced in dimensions -> given as input to RL agent -> outputs 10 
    values -> fed into NN that gives width * height outputs (image transform)
'''

import gym
import numpy as np

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
env = gym.make('MountainCarContinuous-v0')


# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
print(n_actions)
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)
model.learn(total_timesteps=100)
model.save("ddpg_mountain")

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_mountain")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()