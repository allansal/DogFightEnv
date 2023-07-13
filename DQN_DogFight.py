import pickle
import gymnasium as gym
import math
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import pandas as pd

import gym_env

from collections import deque, namedtuple
from itertools import count
from tensordict import TensorDict
from torch import nn, optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyMemmapStorage
from torchvision import transforms as T
import torch.nn.functional as F

# Custom wrappers

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias = None), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

# Define gym environment and apply wrappers
env = gym.make("gym_env/DogFight")
env = SkipFrame(env, skip = 1)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape = 100)
env = gym.wrappers.FrameStack(env, num_stack = 4, lz4_compress = False)

#env = gym.wrappers.FrameStack(env, 4, lz4_compress = True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

episodes_done = 0

# Set up DQN network, layer-by-layer
class DQN(nn.Module):
    def __init__(self, n_state_dim, n_actions):
        super().__init__()
        c, w, h = n_state_dim

        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Forward propagate a dummy input to calculate the flattened size
        dummy_input = torch.zeros(32, c, w, h)
        conv_output = self._forward_conv(dummy_input)
        flattened_size = int(np.prod(conv_output.size()))

        self.fc1 = nn.Linear(flattened_size, 512)  # This size depends on the output of your conv layers
        self.fc2 = nn.Linear(512, n_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        return self.fc2(x)
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = F.relu(self.conv3(x))
#        x = x.view(x.size(0), -1)  # flatten the tensor
#        x = F.relu(self.fc1(x))
#        return self.fc2(x)


#self.online = nn.Sequential(
#            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
#            nn.ReLU(),
#            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#            nn.ReLU(),
#            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
#            nn.ReLU(),
#            nn.Flatten(),
#            nn.Linear(3136, 512),
#            nn.ReLU(),
#            nn.Linear(512, output_dim),
#        )
#
#        self.target = copy.deepcopy(self.online)
#
#        # Q_target parameters are frozen.
#        for p in self.target.parameters():
#            p.requires_grad = False
#        self.
#        super().__init__()
#        c, w, h = n_state_dim
#
#        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
#        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#        self.fc1 = nn.Linear(22*22*64, 512)  # This size depends on the output of your conv layers
#        self.fc2 = nn.Linear(512, n_actions)
#
#    def forward(self, x):
#        x = torch.relu(self.conv1(x))
#        x = torch.relu(self.conv2(x))
#        x = torch.relu(self.conv3(x))
#        x = x.view(x.size(0), -1)  # Flatten the tensor
#        x = torch.relu(self.fc1(x))
#        return self.fc2(x)

# Hyper-parameters for RL training
BATCH_SIZE = 32
GAMMA = 0.99
LR = 2.5e-4
# Eps-greedy algorithm parameters
EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY = 100
# Update rate of target network
TAU = 0.0025

# Get possible # of actions from the environment
n_actions = env.action_space.n
# Reset env to initial state, get initial observations for agent
state, info = env.reset()
print(f"State shape: {state.shape}")
# Get the # of observations of the state (size of input layer)
n_observations = len(state)

# Set up policy & target network w/ proper input and output layer sizes
# Learning target constantly shifts as parameters of the DQN are updated.
# This is a problem since it can cause the learning to diverge.
# Separate target network is used to calc. target Q-value.
# Target has same structure as policy NN, but parameters are frozen.
# Target network updated only occasionally to prevent prevent extreme
# divergence or the agent "forgetting" how to act properly.
#policy_net = DQN(n_observations, n_actions).to(device)
#target_net = DQN(n_observations, n_actions).to(device)
policy_net = DQN(state.shape, n_actions).to(device)
target_net = DQN(state.shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
for p in target_net.parameters():
    p.requires_grad = False
#policy_net.load_state_dict(torch.load("./checkpoints/00999_policy.chkpt"))
#target_net.load_state_dict(torch.load("./checkpoints/00999_target.chkpt"))
#policy_net.eval()
#target_net.eval()

# AdamW optimizer w/ parameters set
optimizer = optim.AdamW(policy_net.parameters(), lr = LR, amsgrad = True)
memory = TensorDictPrioritizedReplayBuffer(
    alpha = 0.65,
    beta = 0.45,
    eps = 1e-6,
    storage = LazyMemmapStorage(
        max_size = 100000,
        device = device
    ),
    batch_size = BATCH_SIZE,
    pin_memory = False
)
#memory.load_state_dict(torch.load("./checkpoints/00999_memory.chkpt"))

# Steps done for eps-greedy algorithm
# As steps grow, make it less likely to choose actions randomly
def select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episodes_done / EPS_DECAY)
    if eps_threshold < sample:
        with torch.no_grad():
            return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)
#            tmp = policy_net(state).max(1)
#            print(f"policy_net(state).max(1) shape: {tmp.shape}")
            return policy_net(state).max(1)
#            return policy_net(state).max(1)[1][0].view(1, 1)
#            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

# Track the durations through the episodes of cartpole, high is better (basically track performance for this environment)
episode_durations = []

# Optimization
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample().to(device)
    states, actions, next_states, rewards, terminations = (batch.get(key) for key in ("state", "action", "next_state", "reward", "terminated"))
    actions = actions.squeeze()
    rewards = rewards.squeeze()
    terminations = terminations.squeeze()
#    print(f"Shapes: {states.shape}, {actions.shape}, {next_states.shape}, {rewards.shape}, {terminations.shape}")
    state_action_values = policy_net(states).gather(1, actions)
#    state_action_values = policy_net(states).gather(1, actions)

    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]

    expected_state_action_values = rewards + (1. - terminations.float()) * GAMMA * next_state_values

    criterion = nn.MSELoss(reduction = "none")
    td_errors = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    weights = batch.get("_weight")
    loss = (weights * td_errors).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 1)
    optimizer.step()

    batch.set("td_error", td_errors)
    memory.update_tensordict_priority(batch)

num_episodes = 10
episode_rewards = []
# Train for the desired # of episodes
i = 0
for i in range(num_episodes):
    ep_step_count = 0
    # Get initial state of episode
    state, info = env.reset()
#    state = np.array(state)
    state = torch.from_numpy(np.array(state)).to(device)
#    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    running_reward = 0
    # Continue until termination
    for t in count():
        # Select action
        action = select_action(state)
        # Get observation, reward, whether we fail or not
        next_state, reward, terminated, truncated, info = env.step(action.item())
        running_reward += reward
        ep_step_count += 1
        reward = torch.tensor([reward], device = device)
        done = terminated or truncated

        if done:
            break

        next_state = torch.from_numpy(np.array(next_state)).to(device)
        state = next_state

        terminated = torch.tensor([terminated], dtype = torch.bool, device = device)
        memory.add(TensorDict({"state": state, "action": action, "next_state": next_state, "reward": reward, "terminated": terminated}, batch_size = []))
        optimize_model()
    
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

        state = next_state

    episodes_done += 1
    episode_rewards.append(running_reward)
    if (i + 1) % 100 == 0 or (i + 1) == num_episodes:
        torch.save(policy_net.state_dict(), f"./checkpoints/{i:05d}_policy.chkpt")
        torch.save(target_net.state_dict(), f"./checkpoints/{i:05d}_target.chkpt")
        torch.save(memory.state_dict(), f"./checkpoints/{i:05d}_memory.chkpt")

    print(f"Episode {i:5d} ended, reward: {running_reward}")

s = pd.Series(episode_rewards)
s_ma = s.rolling(10).mean()
print("Complete")
fig, ax = plt.subplots()
ax.plot(s, label = "Raw Rewards")
ax.plot(s_ma, label = "Rewards (moving average)")
ax.legend()
plt.xlabel("Episode Number")
plt.ylabel("Episode Reward")
plt.title("RL Reward Across Training Episodes")
plt.savefig("./episode_reward_plot.png")
print(f"Average reward: {sum(episode_rewards) / len(episode_rewards)}")
print(f"Max reward during Episode {episode_rewards.index(max(episode_rewards))}: {max(episode_rewards)}")
