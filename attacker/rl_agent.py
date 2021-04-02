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
from SSD.ssd.modeling.detector.ssd_detector import SSDDetector
from SSD.ssd.config import cfg
from SSD.ssd.utils.logger import setup_logger
from SSD.ssd.utils import dist_util
from SSD.ssd.utils.dist_util import synchronize
import math
# CONSTANTS

TAU = 0.001
REPLAY_BUFFER_SIZE = 10**6
EPSILON_DECAY = 0.001
TRAIN_STEPS = 5
IN_CHANNELS = 3
IMAGE_SHAPE = (32,32)
RL_BATCH_SIZE = 16  # we train the agents over RL_BATCH_SIZE images at a time (the reward is the average MAP degradation in the batch)
KERNEL_SIZE = 5
# The RL agent takes in an image, and then applies a perturbation to the image, and observes the reward from that.

class DeepActorCriticNetwork(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate: int,
                 input_features: int,
                 output_features: int,
                 hidden_features,
                 l2_decay: int,
                 name: str,
                 checkpoint_dir: str):
        super().__init__()

        self.checkpoint_dir = os.path.join(checkpoint_dir, name + '_ddpg_actor')
        self.learning_rate = learning_rate
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.hidden_layers = nn.ModuleList()
        self.l2_decay = l2_decay

        self.initialize_model()  # abstract method

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    @abc.abstractmethod
    def initialize_model(self):
        pass


class Actor(DeepActorCriticNetwork):
    """
    Selects the action for a specific state
    """
    def __init__(self, learning_rate: int,
                 input_features: int,
                 output_features: int,
                 hidden_features,
                 l2_decay: int,
                 name: str,
                 checkpoint_dir: str):
        super().__init__(learning_rate, input_features, output_features, hidden_features, l2_decay, name, checkpoint_dir)

    def initialize_model(self):
        self.input_layer = nn.Sequential(
            nn.Linear(
                in_features=self.input_features,
                out_features=self.hidden_features[0]
            ),
            #nn.BatchNorm1d(self.hidden_features[0]),
            nn.ReLU(),
        )

        f1 = 1 / (np.sqrt(self.input_layer[0].weight.data.size()[0]))
        nn.init.uniform(self.input_layer[0].weight.data, -f1, f1)
        nn.init.uniform(self.input_layer[0].bias.data, -f1, f1)
        for i in range(1, len(self.hidden_features)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.hidden_features[i - 1],
                        out_features=self.hidden_features[i]
                    ),
                    #nn.BatchNorm1d(self.hidden_features[i]),
                    nn.ReLU()
                )
            )
            fn = 1 / (np.sqrt(self.hidden_layers[i-1][0].weight.data.size()[0]))
            nn.init.uniform(self.input_layer[0].weight.data, -fn, fn)

        self.output_layer = nn.Sequential(  # the Q-value estimator
            nn.Linear(
                in_features=self.hidden_features[-1],
                out_features=self.output_features  # this is probably 1
            ),
            nn.Tanh()  # normalizes output to -1, 1
        )

        f_output = 1 / (np.sqrt(self.output_layer[0].weight.data.size()[0]))
        nn.init.uniform_(self.output_layer[0].weight.data, -f_output, f_output)
        nn.init.uniform_(self.output_layer[0].bias.data, -f_output, f_output)

    def forward(self, x):
        v = self.input_layer(x)
        for layer in self.hidden_layers:
            v = layer(v)
        v = self.output_layer(v)
        return v

class Critic(nn.Module):
    """
    Criticizes action
    """
    def __init__(self, learning_rate: int,
                 input_features: int,
                 output_features: int,
                 hidden_features,
                 l2_decay: int,
                 name: str,
                 checkpoint_dir: str):
        super().__init__()

    def initialize_model(self):
        self.input_layer = nn.Sequential(
            nn.Linear(
                in_features=self.input_features,
                out_features=self.hidden_features[0]
            ),
            nn.BatchNorm1d(self.hidden_features[0]),
            nn.ReLU(),
        )

        f1 = 1 / (np.sqrt(self.input_layer.weight.data.size()[0]))
        nn.init.uniform(self.input_layer[0].weight.data, -f1, f1)
        nn.init.uniform(self.input_layer[0].bias.data, -f1, f1)
        for i in range(1, len(self.hidden_features)):
            self.hidden_layers.add(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.hidden_features[i - 1],
                        out_features=self.hidden_features[i]
                    ),
                    #nn.BatchNorm1d(self.hidden_features[i]),
                    nn.ReLU()
                )
            )
            fn = 1 / (np.sqrt(self.hidden_layers[i][0].weight.data.size()[0]))
            nn.init.uniform(self.input_layer[0].weight.data, -fn, fn)

        self.output_layer = nn.Sequential(  # the Q-value estimator
            nn.Linear(
                in_features=self.hidden_features[-1],
                out_features=self.output_features  # this is probably 1
            ),
            nn.Tanh()  # normalizes output to -1, 1
        )

        self.action_value = nn.Sequential(
            nn.Linear(
                in_features=self.output_features,
                out_features=self.output_features
            ),
            nn.ReLU()
        )

        f_output = 1 / (np.sqrt(self.output_layer.weight.data.size()[0]))
        nn.init.uniform(self.output_layer[0].weight.data, -f_output, f_output)
        nn.init.uniform(self.output_layer[0].bias.data, -f_output, f_output)

    def forward(self, state, action):
        state_value = self.input_layer(state)
        for layer in self.hidden_layers:
            state_value = layer(state_value)

        action_value = self.action_value(action)
        state_action_value = torch.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.output_layer(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_dir)
        self.load_state_dict(torch.load(self.checkpoint_dir))


def create_target_network(net):
    return copy.deepcopy(net)


class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size)
        self.buffer_counter = 0

        self.old_state_buffer = np.zeros((self.buffer_size, *input_shape))
        self.new_state_buffer = np.zeros((self.buffer_size, *input_shape))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.terminal_buffer = np.zeros(self.buffer_size, dtype=np.float32)

    def record_transition(self, old_state, new_state, action, reward, done):
        # In DDPG, the oldest samples are discarded when the buffer is full
        index = self.buffer_counter % self.buffer_size

        self.old_state_buffer[index] = old_state
        self.new_state_buffer[index] = new_state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.terminal_buffer[index] = 1 - int(done)
        self.buffer_counter += 1

    def sample(self, n):
        '''
        Samples a batch of memories from the replay buffer
        :param n: the batch size
        :return: old states, new states, rewards, actions, terminal
        '''
        max_idx = min(self.buffer_counter, self.buffer_size)
        index = np.random.choice(max_idx, n)
        old_states = self.old_state_buffer[index]
        new_states = self.new_state_buffer[index]
        rewards = self.reward_buffer[index]
        actions = self.action_buffer[index]
        terminal = self.terminal_buffer[index]

        return old_states, new_states, rewards, actions, terminal


class RandomProcess:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return random.random()


class RLAgent:
    '''
    DDPG RL attacking agent
    Is responsible for attacking a single pixel
    '''
    def __init__(self,
                 environment,
                 in_channels,
                 tau=TAU,
                 replay_buffer_size=REPLAY_BUFFER_SIZE,
                 epsilon_decay=EPSILON_DECAY,
                 checkpoint_dir="",
                 kernel_size=5,
                 ):
        """
        :param environment:
        :param input_shape:
        :param in_channels:
        :param tau:
        :param replay_buffer_size:
        :param epsilon_decay:
        :param checkpoint_dir:
        :param kernel_size: size of the region to consider when attacking. is a config parameter that is trained on
        :param eval_region_size:
        """
        self.actor_net = Actor(input_features=kernel_size ** 2, output_features=1, hidden_features=[32, 32], learning_rate=1, l2_decay=1, name="actor", checkpoint_dir="")
        self.critic_net = Critic(input_features=kernel_size ** 2, output_features=1, hidden_features=[32, 32], learning_rate=1, l2_decay=1, name="actor", checkpoint_dir="")
        self.actor_target = create_target_network(self.actor_net)
        self.critic_target = create_target_network(self.critic_net)
        self.tau = tau
        self.replay_buffer = ReplayBuffer(replay_buffer_size, [kernel_size ** 2], in_channels)
        self.noise = RandomProcess()
        self.environment = environment

        self.kernel_size = kernel_size

    def after_step(self, old_state, new_state, reward, action, done=False):
        '''
        Needs to be called after the final reward matrix of the image has been computed, for each modified pixel

        :return:
        '''
        self.replay_buffer.record_transition(old_state, new_state, action, reward, done)

    def choose_action(self, state) -> int:
        '''
        Gives the highest-valued action or a random exploring action
        State is a kernel observation. So not the whole image.
        :return:
        '''
        action = self.actor_net(state) + self.noise()
        return action

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass


class Environment:

    def __init__(self, image_dim, obj_detector, dataloader, batch_size=RL_BATCH_SIZE, in_channels=IN_CHANNELS, train_steps=TRAIN_STEPS):
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.agent = self.initialize_rl_agent()
        self.train_steps = train_steps
        self.obj_detector = obj_detector
        self.batch_size = batch_size
        self.dataloader = dataloader

        self.image = np.zeros(image_dim, dtype=np.float32)  # current image TODO make tuple with correct labels
        self.attack_sequence = []

    def initialize_rl_agent(self):
        return RLAgent(environment=self, kernel_size=KERNEL_SIZE, in_channels=self.in_channels)

    def calculate_map(self):
        #result = self.obj_detector(self.image)
        #TODO call normal ap func, return that
        return 0

    def image_to_state(self):
        return self.image

    def next_image(self):
        # self.image = self.dataloader.next_image() # TODO
        pass

    def get_state(self, pixel, kernel_size):
        dist = kernel_size // 2
        left_boundary = pixel[0] - dist
        right_boundary = pixel[0] + dist
        top_boundary = pixel[1] - dist
        bottom_boundary = pixel[1] + dist

        kernel = np.zeros((kernel_size, kernel_size))
        for x in range(left_boundary, right_boundary):
            for y in range(top_boundary, bottom_boundary):
                if x < 0 or x >= self.image.shape[0] or y < 0 or y >= self.image.shape[1]:
                    val = 0  # OUT OF BOUNDS
                else:
                    val = self.image[x][y]
                kernel[x - left_boundary][y - top_boundary] = val
        return torch.from_numpy(kernel.flatten()).float()

    def attack_pixel(self, action, pixel):
        self.image[pixel] = min(max(0, self.image[pixel] + action), 1)

    def attack_image(self):
        """
        Performs the attack on the image.
        :return:
        """
        # divide image into regions
        # call attack_region() on each region

        regions = []
        i = 0
        pixels_per_region = 1
        pixel_amount = self.image.shape[0] * self.image.shape[1]
        region = []
        while i < pixel_amount:


            row = i // self.image.shape[1]
            col = i % self.image.shape[0]
            i += 1
            region.append((row, col))
            if i >= pixel_amount or len(region) >= pixels_per_region:
                regions.append(region)
                region = []

        ap = None
        for region in regions:
            ap = self.attack_region(region, ap)

        # whole image is attacked. Now, we give the agent the delayed rewards
        for operation in self.attack_sequence:
            for i in range(len(operation["region"])):
                self.agent.after_step(operation["old_states"][i], operation["new_states"][i], operation["actions"][i], operation["rewards"][i])



    def distance_func(self, pixel, other_pixel):
        return abs(other_pixel[0] - pixel[0]) + abs(other_pixel[1] - pixel[1])  # TODO inverse

    def propagate_reward_spatially(self, last_operation):
        for p, pixel in enumerate(last_operation["region"]):
            for i, operation in enumerate(self.attack_sequence):
                for j in range(len(operation["region"])):
                    other_pixel = operation["region"][j]
                    dist = self.distance_func(pixel, other_pixel)
                    self.attack_sequence[i]["rewards"][j] += last_operation["rewards"][p] * dist

    def attack_region(self, region, ap_before) -> int:
        # for each pixel in region:
        # get agent's action for that pixel with the kernel values as input
        # call attack_image on that pixel
        # after region is attacked, call the object detector with the current image state. calculate AP.
        # set the reward of each of the corresponding pixels, logging the input state as well as the action
        # propagate the reward spatially to previously modified pixels
        old_states = []
        new_states = []
        actions = []
        if ap_before is None:
            ap_before = self.calculate_map()
        for pixel in region:
            old_state = self.get_state(pixel, self.agent.kernel_size)
            old_states.append(old_state)
            action = self.agent.choose_action(old_state)
            self.attack_pixel(action, pixel)
            new_state = self.get_state(pixel, self.agent.kernel_size)
            new_states.append(new_state)
            actions.append(action)

        ap_after = self.calculate_map()
        reward = ap_after - ap_before
        operation = {
                "region": region,
                "rewards": [reward for _ in region],
                "old_states": old_states,
                "new_states": new_states,
                "actions": actions
            }

        self.propagate_reward_spatially(operation)
        self.attack_sequence.append(operation)
        return ap_after

    def train(self):
        step = 0
        while step < self.train_steps:
            self.attack_sequence = []
            self.next_image()
            self.attack_image()

            step += 1


# Need to init an obj detectyor, as well as a dataloader

def main():
    parser = argparse.ArgumentParser(description='SSD Evaluation on VOC and COCO dataset.')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

'''
    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
'''


cfg = main()
env = Environment(image_dim=(32,32), obj_detector=SSDDetector(cfg=cfg), dataloader=None)
env.train()





