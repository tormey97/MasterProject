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
from SSD.ssd.config import cfg
from SSD.ssd.utils.logger import setup_logger
from SSD.ssd.utils import dist_util
from SSD.ssd.utils.dist_util import synchronize
from SSD.ssd.data.build import make_data_loader
from SSD.ssd.data.datasets.evaluation.voc.eval_detection_voc import *
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

'''
#TODO what is left:
RL agent needs to actually learn. Specifically look into how the Critic is supposed to work
Need to load images from dataset
Need to save models as checkpoints and load them
Need to calculate AP for SSD
give reward correctly
some variables are hardcoded, should be configurable
'''
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
        self.to("cuda")

        self.initialize_model()  # abstract method

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to("cuda")

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
            nn.init.uniform(self.input_layer[0].bias.data, -fn, fn)

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

class Critic(DeepActorCriticNetwork):
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
        super().__init__(learning_rate, input_features, output_features, hidden_features, l2_decay, name, checkpoint_dir)

    def initialize_model(self):
        self.input_layer = nn.Sequential(
            nn.Linear(
                in_features=self.input_features,
                out_features=self.hidden_features[0]
            ),
            nn.BatchNorm1d(self.hidden_features[0]),
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
                    nn.BatchNorm1d(self.hidden_features[i]),
                    nn.ReLU()
                )
            )
            fn = 1 / (np.sqrt(self.hidden_layers[i - 1][0].weight.data.size()[0]))
            nn.init.uniform(self.input_layer[0].weight.data, -fn, fn)
            nn.init.uniform(self.input_layer[0].bias.data, -fn, fn)

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
                out_features=self.hidden_features[-1]
            ),
            nn.ReLU()
        )

        f_output = 1 / (np.sqrt(self.output_layer[0].weight.data.size()[0]))
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
        self.action_buffer[index] = action.cpu().detach().numpy()
        self.reward_buffer[index] = reward
        self.terminal_buffer[index] = bool(1 - int(done))
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
        return old_states, actions, rewards, new_states, terminal


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        return 0 #TODO

        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

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
                 batch_size=RL_BATCH_SIZE
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
        input_features = (kernel_size ** 2) * IN_CHANNELS
        self.actor_net = Actor(input_features=input_features, output_features=IN_CHANNELS, hidden_features=[32, 32], learning_rate=1, l2_decay=1, name="actor", checkpoint_dir="")
        self.critic_net = Critic(input_features=input_features, output_features=IN_CHANNELS, hidden_features=[32, 32], learning_rate=1, l2_decay=1, name="actor", checkpoint_dir="")
        self.actor_target = create_target_network(self.actor_net)
        self.critic_target = create_target_network(self.critic_net)
        self.tau = tau
        self.gamma = 0.99 #TODO
        self.replay_buffer = ReplayBuffer(replay_buffer_size, [input_features], in_channels)
        self.noise = OUActionNoise(mean=0, std_deviation=1)
        self.environment = environment
        self.kernel_size = kernel_size
        self.batch_size=batch_size
        self.update_network_parameters(tau=1)

    def after_step(self, old_state, new_state, reward, action, done=False):
        '''
        Needs to be called after the final reward matrix of the image has been computed, for each modified pixel

        :return:
        '''
        self.replay_buffer.record_transition(old_state, new_state, action, reward, done)
        self.learn()

    def choose_action(self, state):
        '''
        Gives the highest-valued action or a random exploring action
        State is a kernel observation. So not the whole image.
        :return:
        '''
        action = self.actor_net(state) + self.noise()
        return action

    def learn(self):
        if self.replay_buffer.buffer_counter < self.batch_size:
            return

        states, actions, rewards, states_, done = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.actor_net.device)
        states_ = torch.tensor(states_, dtype=torch.float).to(self.actor_net.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor_net.device)
        done = torch.tensor(done).to(self.actor_net.device)

        target_actions = self.actor_target.forward(states_)
        critic_value_ = self.critic_target.forward(states_, target_actions)
        critic_value = self.critic_net.forward(states, actions)
        '''
        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)
        '''

        rewards = np.tile(rewards, (IN_CHANNELS,1)).T
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor_net.device)
        target = rewards + self.gamma * critic_value_
        target = target.T.view(self.batch_size, IN_CHANNELS)

        self.critic_net.train()
        self.critic_net.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_net.optimizer.step()

        self.actor_net.optimizer.zero_grad()
        actor_loss = -self.critic_net.forward(states, self.actor_net.forward(states))
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor_net.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor_net.named_parameters()
        critic_params = self.critic_net.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_params = self.critic_target.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                                      (1 - tau) * target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                     (1 - tau) * target_actor_state_dict[name].clone()

        self.critic_target.load_state_dict(critic_state_dict, strict=False)
        self.actor_target.load_state_dict(actor_state_dict, strict=False)
        self.critic_target.load_state_dict(critic_state_dict, strict=False)
        self.actor_target.load_state_dict(actor_state_dict, strict=False)

    def save_checkpoint(self):
        self.actor_net.save_checkpoint()
        self.critic_net.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_checkpoint(self):
        self.actor_net.load_checkpoint()
        self.critic_net.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic_target.load_checkpoint()


class Environment:

    def __init__(self, image_dim, obj_detector, cfg, batch_size=RL_BATCH_SIZE, in_channels=IN_CHANNELS, train_steps=TRAIN_STEPS):
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.agent = self.initialize_rl_agent()
        self.train_steps = train_steps
        self.obj_detector = obj_detector
        self.obj_detector.eval()
        self.batch_size = batch_size
        self.dataloader = make_data_loader(cfg)

        self.targets = None #targets of current image
        self.annotation = None # annotation of current image
        self.image = None  # current image TODO make tuple with correct labels
        self.attack_sequence = []
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.obj_detector.cuda()

    def initialize_rl_agent(self):
        return RLAgent(environment=self, kernel_size=KERNEL_SIZE, in_channels=self.in_channels)

    def calculate_map(self):
        preds = self.obj_detector(torch.Tensor([self.image.detach().numpy()]).to("cuda"), targets=self.targets) #TODO device

        prec, rec = calc_detection_voc_prec_rec([preds[0]["boxes"].detach().to("cpu").numpy()],
                                                [preds[0]["labels"].detach().to("cpu").numpy()],
                                                [preds[0]["scores"].detach().to("cpu").numpy()],
                                                [self.annotation[0]],
                                                [self.annotation[1]],
                                                None,
                                                iou_thresh=0.5)

        with torch.no_grad():
            ap = calc_detection_voc_ap(prec, rec, use_07_metric=False)

        return np.nan_to_num(ap).mean()

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

        kernel = np.zeros((kernel_size, kernel_size, IN_CHANNELS))
        # TODO make FASTER
        for x in range(left_boundary, right_boundary):
            for y in range(top_boundary, bottom_boundary):
                if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[2]:
                    val = np.zeros(IN_CHANNELS)  # OUT OF BOUNDS
                else:
                    val = self.image[:, x, y].cpu().detach().numpy()
                kernel[x - left_boundary][y - top_boundary] = val
        return torch.from_numpy(kernel.flatten()).float().to("cuda")

    def attack_pixel(self, action, pixel):
        diffs = []
        for i, action_ in enumerate(action):
            old = self.image[i][pixel].item()
            new_value = min(max(-122, self.image[i][pixel] + (action_ * 254) - 122), 122)
            self.image[i][pixel] = new_value  # TODO normalize -123 +122
            new = self.image[i][pixel].item()
            diffs.append(new_value - old)
        return sum(diffs)

    def attack_image(self):
        """
        Performs the attack on the image.
        :return:
        """
        # divide image into regions
        # call attack_region() on each region

        regions = []
        i = 0
        pixels_per_region = 30
        pixel_amount = self.image.shape[1] * self.image.shape[2]
        region = []
        while i < pixel_amount:
            row = i // self.image.shape[1]
            col = i % self.image.shape[2]
            i += 1
            region.append((row, col))
            if i >= pixel_amount or len(region) >= pixels_per_region:
                regions.append(region)
                region = []

        ap = None
        for region in regions:
            ap = self.attack_region(region, ap)
            break #TODO remove

        # whole image is attacked. Now, we give the agent the delayed rewards
        for operation in self.attack_sequence:
            for i in range(len(operation["region"])):
                self.agent.after_step(old_state=operation["old_states"][i], new_state=operation["new_states"][i], action=operation["actions"][i], reward=operation["rewards"][i])

    def distance_func(self, pixel, other_pixel):
        return abs(other_pixel[0] - pixel[0]) + abs(other_pixel[1] - pixel[1])  # TODO inverse

    def propagate_reward_spatially(self, last_operation): # TODO needs to be faster
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
        differences = []
        print("Beginning attack on region")

        if ap_before is None:
            print("Calcing map")

            ap_before = self.calculate_map()
            print("Map calced")

        for pixel in region:
            time1 = time.time()

            old_state = self.get_state(pixel, self.agent.kernel_size)
            old_states.append(old_state.cpu())
            action = self.agent.choose_action(old_state)
            difference = self.attack_pixel(action, pixel)
            new_state = self.get_state(pixel, self.agent.kernel_size)
            new_states.append(new_state.cpu())
            actions.append(action.cpu())
            differences.append(difference)
            time2 = time.time() - time1

        print("Attack finished")
        print("Calcing map")
        ap_after = self.calculate_map()
        print("Map calced")
        reward = ap_after - ap_before
        kernel_center = self.agent.kernel_size // 2
        operation = {
                "region": region,
                "rewards": [reward for _ in range(len(region))],
                "old_states": old_states,
                "new_states": new_states,
                "actions": actions
            }

        self.propagate_reward_spatially(operation)
        for i in range(len(region)):
            operation["rewards"][i] -= abs(differences[i])
        self.attack_sequence.append(operation)

        return ap_after

    def train(self):
        step = 0
        lngth = len(self.dataloader)
        for iteration, (images, targets, scores) in enumerate(self.dataloader):
            print(iteration)
            self.annotation = self.dataloader.dataset.datasets[0].get_annotation(iteration)
            self.targets = targets
            self.image = images[0] # The batch size is 1.
            self.attack_sequence = []
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


if __name__=="__main__":
    cfg = main()
    env = Environment(image_dim=(32,32), obj_detector=SSDDetector(cfg=cfg), cfg=cfg)
    env.train()





