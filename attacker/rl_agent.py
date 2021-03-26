import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import abc
# CONSTANTS

TAU = 0.001
REPLAY_BUFFER_SIZE = 10**6
EPSILON_DECAY = 0.001
TRAIN_STEPS = 10000
IN_CHANNELS = 3
IMAGE_SHAPE = (32,32)
RL_BATCH_SIZE = 16  # we train the agents over RL_BATCH_SIZE images at a time (the reward is the average MAP degradation in the batch)

# The RL agent takes in an image, and then applies a perturbation to the image, and observes the reward from that.

class DeepActorCriticNetwork(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, learning_rate: int,
                 input_features: int,
                 output_features: int,
                 hidden_features: list[int],
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
                 hidden_features: list[int],
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
                    nn.BatchNorm1d(self.hidden_features[i]),
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

        f_output = 1 / (np.sqrt(self.output_layer.weight.data.size()[0]))
        nn.init.uniform(self.output_layer[0].weight.data, -f_output, f_output)
        nn.init.uniform(self.output_layer[0].bias.data, -f_output, f_output)

    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    """
    Criticizes action
    """
    def __init__(self, learning_rate: int,
                 input_features: int,
                 output_features: int,
                 hidden_features: list[int],
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
                    nn.BatchNorm1d(self.hidden_features[i]),
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
    return net.deepcopy()


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
                 coords,
                 input_shape,
                 in_channels,
                 tau=TAU,
                 replay_buffer_size=REPLAY_BUFFER_SIZE,
                 epsilon_decay=EPSILON_DECAY,
                 checkpoint_dir=""
                 ):
        self.actor_net = Actor(input_features=input_shape, output_features=None, hidden_features=None)
        self.critic_net = Critic(input_features=input_shape, output_features=None, hidden_features=None)
        self.actor_target = create_target_network(self.actor_net)
        self.critic_target = create_target_network(self.critic_net)
        self.tau = tau
        self.last_observation = None
        self.image_dim = input_shape
        self.replay_buffer = ReplayBuffer(replay_buffer_size, input_shape, in_channels)
        self.noise = RandomProcess()
        self.environment = environment
        self.coords = coords


    def after_step(self, old_state, new_state, reward):
        '''
        Gets the resulting MAP from attacking the detector
        :return:
        '''
        self.replay_buffer.record_transition(old_state, new_state, reward)
        pass

    def choose_action(self, state) -> int:
        '''
        Gives the highest-valued action or a random exploring action
        :return:
        '''
        action = self.actor_net(state) + self.noise()
        return action

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass



class Environment:

    def __init__(self, image_dim, batch_size=RL_BATCH_SIZE, in_channels=IN_CHANNELS, train_steps=TRAIN_STEPS):
        self.image_dim = image_dim
        self.in_channels = in_channels
        self.agents = [[self.initialize_rl_agent((x, y)) for x in image_dim[0]] for y in image_dim[1]]
        self.train_steps = train_steps
        self.obj_detector = None
        self.image = np.zeros(image_dim)  # current image
        self.batch_size = batch_size
        self.dataloader = None

    def initialize_rl_agent(self, coords):
        return RLAgent(environment=self, coords=coords, input_shape=self.image_dim, in_channels=self.in_channels)

    def apply_perturbation(self, coords: tuple[int, int], action):
        self.image[coords] += action

    def calculate_map(self):
        pass

    def image_to_state(self):
        return self.image

    def next_image(self):
        self.image = self.dataloader.next_image()
        return np.copy(self.image)

    def train(self):
        step = 0
        while step < self.train_steps:
            old_state = self.next_image()
            map_before = self.calculate_map()
            for x in range(len(self.agents)):
                for y in range(len(self.agents[x])):
                    agent = self.agents[x][y]
                    action = agent.choose_action(self.image)
                    self.apply_perturbation((x, y), action)
            # image is perturbed. now calculate MAP

            map_after = self.calculate_map()
            reward = map_before - map_after
            for agent in self.agents:
                agent.after_step(old_state, self.image_to_state(), reward)






