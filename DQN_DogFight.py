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
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage

# Define gym environment
env = gym.make("gym_env/DogFight")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

episodes_done = 0

# Set up DQN network, layer-by-layer
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, n_actions)

    # Forward pass through NN
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        return self.layer4(x)

# Hyper-parameters for RL training
BATCH_SIZE = 128
GAMMA = 0.99
LR = 5e-4
# Eps-greedy algorithm parameters
EPS_START = 1.00
EPS_END = 0.05
EPS_DECAY = 100
# Update rate of target network
TAU = 0.005

# Get possible # of actions from the environment
n_actions = env.action_space.n
# Reset env to initial state, get initial observations for agent
state, info = env.reset()
# Get the # of observations of the state (size of input layer)
n_observations = len(state)

# Set up policy & target network w/ proper input and output layer sizes
# Learning target constantly shifts as parameters of the DQN are updated.
# This is a problem since it can cause the learning to diverge.
# Separate target network is used to calc. target Q-value.
# Target has same structure as policy NN, but parameters are frozen.
# Target network updated only occasionally to prevent prevent extreme
# divergence or the agent "forgetting" how to act properly.
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
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
    storage = LazyTensorStorage(
        max_size = 125 * 60 * env.metadata["render_fps"],
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
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

# Track the durations through the episodes of cartpole, high is better (basically track performance for this environment)
episode_durations = []

# Optimization
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample()
    states, actions, next_states, rewards, terminations = (batch.get(key) for key in ("state", "action", "next_state", "reward", "terminated"))
    states = torch.cat([s for s in batch["state"]])
    actions = torch.cat([s for s in batch["action"]])
    next_states = torch.cat([s for s in batch["next_state"]])
    rewards = torch.cat([s for s in batch["reward"]])
    terminations = torch.cat([s for s in batch["terminated"]])

    state_action_values = policy_net(states).gather(1, actions)

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

num_episodes = 1000
episode_rewards = []
# Train for the desired # of episodes
i = 0
for i in range(num_episodes):
    ep_step_count = 0
    # Get initial state of episode
    state, info = env.reset()
    state = torch.tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
    shooting_transitions = []
    shooting_flags = []
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

        next_state = torch.tensor(next_state, dtype = torch.float32, device = device).unsqueeze(0)

        terminated = torch.tensor([terminated], dtype = torch.bool, device = device)
        # Add experience to local memory if it is a shooting state
        # otherwise push to the global memory
        if (
            (info["shoot_id"] is not None) or
            (len(info["hit_ids"]) > 0) or
            (len(info["miss_ids"]) > 0)
        ):
            shooting_transitions.append((state, action, next_state, reward, terminated))
            shooting_flags.append(info)
        else:
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

    for x, transition in reversed(list(enumerate(shooting_transitions))):
        for idx, hit_missile_id in reversed(list(enumerate(shooting_flags[x]["hit_ids"]))):
            y = x - 1
            while y >= 0:
                if shooting_flags[y]["shoot_id"] == hit_missile_id:
                    old_reward = shooting_transitions[y][3].item()
                    new_reward = old_reward + shooting_flags[x]["hit_rewards"][idx]
                    running_reward += shooting_flags[x]["hit_rewards"][idx]
                    shooting_transitions[y] = (
                        shooting_transitions[y][0],
                        shooting_transitions[y][1],
                        shooting_transitions[y][2],
                        torch.tensor([new_reward], device = device)
                    )
                y -= 1
        for idx, miss_missile_id in reversed(list(enumerate(shooting_flags[x]["miss_ids"]))):
            y = x - 1
            while y >= 0:
                if shooting_flags[y]["shoot_id"] == miss_missile_id:
                    old_reward = shooting_transitions[y][3].item()
                    new_reward = old_reward + shooting_flags[x]["miss_rewards"][idx]
                    running_reward += shooting_flags[x]["miss_rewards"][idx]
                    shooting_transitions[y] = (
                        shooting_transitions[y][0],
                        shooting_transitions[y][1],
                        shooting_transitions[y][2],
                        torch.tensor([new_reward], device = device)
                    )
                y -= 1
        memory.add(TensorDict({"state": transition[0], "action": transition[1], "next_state": transition[2], "reward": transition[3], "terminated": transition[4]}, batch_size = []))
        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

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
