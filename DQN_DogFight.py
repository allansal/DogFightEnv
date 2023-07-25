import argparse
import copy
import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import random
import sys
import torch

import gym_env

from itertools import count
from tensordict import TensorDict
from torch import nn, optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, LazyTensorStorage

# Q-value decomposed DQN
class DecompDQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_reward_components):
        super(DecompDQN, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, n_actions)
            ) for _ in range(n_reward_components)
        ])

    def forward(self, x):
        x = self.trunk(x)
        x = torch.stack([branch(x) for branch in self.branches])
        return x

def main():
    base_dir = "./checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()

    # Create checkpoints folder if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    if not args.checkpoint_dir:
        subdir = datetime.datetime.now().strftime("%m%d%H%M%S")
    else:
        subdir = args.checkpoint_dir
    os.makedirs(f"{base_dir}/{subdir}", exist_ok = True)
    print(f"Created checkpoint directory: {base_dir}/{subdir}")

    # Define gym environment
    render_mode = "human" if args.render else None
    env = gym.make("gym_env/DogFight", render_mode = render_mode)

    state, info = env.reset()
    n_actions = env.action_space.n
    n_observations = len(state)
    n_reward_components = len(info["dreward"])

    # Set up DQN network
    policy_net = DecompDQN(n_observations, n_actions, n_reward_components)
    target_net = DecompDQN(n_observations, n_actions, n_reward_components)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.to(device)
    target_net.to(device)

    # Prioritized experience replay memory buffer
    memory = TensorDictPrioritizedReplayBuffer(
        alpha = args.per_alpha,
        beta = args.per_beta,
        eps = args.per_eps,
        storage = LazyTensorStorage(
            max_size = args.memory_size,
            device = device
        ),
        batch_size = args.batch_size,
        pin_memory = True if torch.cuda.is_available() else False
    )

    start = 0
    num_episodes = args.num_episodes
    episode_rewards = []
    if args.load_checkpoint is not None:
        checkpoint = args.load_checkpoint
        policy_net.load_state_dict(torch.load(f"{base_dir}/{subdir}/{checkpoint}_policy.pt"))
        print(f"Loaded policy_net checkpoint:  {base_dir}/{subdir}/{checkpoint}_policy.pt")
        target_net.load_state_dict(torch.load(f"{base_dir}/{subdir}/{checkpoint}_target.pt"))
        print(f"Loaded target_net checkpoint:  {base_dir}/{subdir}/{checkpoint}_target.pt")
        memory.load_state_dict(torch.load(f"{base_dir}/{subdir}/{checkpoint}_memory.pt"))
        print(f"Loaded memory checkpoint:      {base_dir}/{subdir}/{checkpoint}_memory.pt")
        with open(f"{base_dir}/{subdir}/{checkpoint}_rewards", "rb") as file:
            episode_rewards = pickle.load(file)
            print(f"Loaded reward list checkpoint: {base_dir}/{subdir}/{checkpoint}_rewards")
        start = checkpoint
        num_episodes += checkpoint
    if args.evaluate:
        policy_net.eval()
        target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr = args.lr, amsgrad = True)

    i = start
    j = 0
    eps_min = args.eps_min
    eps_max = args.eps_max
    eps_decay = (eps_min / eps_max) ** (1. / (num_episodes - start))
    while i < num_episodes:
        state, info = env.reset()
        state = torch.tensor(state, device = device, dtype = torch.float32).unsqueeze(0)
        running_reward = 0
        if not args.evaluate:
            eps = max(args.eps_min, eps_max * eps_decay ** i)
        else:
            eps = args.eval_eps
        print("--------------------------------------------------------------------------------")
        if j >= args.exploration_episodes:
            print(f"Episode {i:6d} / {num_episodes:6d} started, epsilon: {eps}")
        else:
            print(f"Pure exploration episode {j:6d} / {args.exploration_episodes:6d} started")

        shooting_transitions = []
        shooting_flags = []
        for t in count():
            # Epsilon greedy
            sample = random.random()
            if sample > eps:
                with torch.no_grad():
                    action = sum(policy_net(state)).max(dim = 1)[1].view(1, 1)
            else:
                action = torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

            next_state, reward, terminated, truncated, info = env.step(action.item())
            running_reward += reward
            dreward = info["dreward"]
            dreward = torch.tensor([dreward], device = device)
            next_state = torch.tensor(next_state, device = device, dtype = torch.float32).unsqueeze(0)
            done = terminated or truncated
            terminated = torch.tensor([terminated], device = device, dtype = torch.bool)

            transition = TensorDict(
                {
                    "state"      : state,
                    "action"     : action,
                    "next_state" : next_state,
                    "dreward"    : dreward,
                    "terminated" : terminated
                },
                batch_size = []
            )

            if info["shoot_act"]:
                shooting_transitions.append(transition)
                shooting_flags.append(info)
            elif not args.evaluate:
                memory.add(transition)
                if j >= args.exploration_episodes:
                    optimize_model(policy_net, target_net, optimizer, args.gamma, memory, n_reward_components)
                    update_target(policy_net, target_net, args.tau)

            if done:
                break

            state = next_state

        for x, transition in reversed(list(enumerate(shooting_transitions))):
            for idx, hit_missile_id in reversed(list(enumerate(shooting_flags[x]["hit_ids"]))):
                y = x - 1
                while y >= 0:
                    if shooting_flags[y]["shoot_id"] == hit_missile_id:
                        new_reward = torch.tensor(shooting_flags[x]["hit_rewards"][idx], device = device)
                        ind = shooting_flags[x]["dhit_ind"][idx]
                        shooting_transitions[y]["dreward"][0, ind] = new_reward
                        running_reward += new_reward.item()
                    y -= 1
            for idx, miss_missile_id in reversed(list(enumerate(shooting_flags[x]["miss_ids"]))):
                y = x - 1
                while y >= 0:
                    if shooting_flags[y]["shoot_id"] == miss_missile_id:
                        new_reward = torch.tensor(shooting_flags[x]["miss_rewards"][idx], device = device)
                        ind = shooting_flags[x]["dmis_ind"][idx]
                        shooting_transitions[y]["dreward"][0, ind] = new_reward
                        running_reward += new_reward.item()
                    y -= 1
            if not args.evaluate:
                memory.add(transition)
                if j >= args.exploration_episodes:
                    optimize_model(policy_net, target_net, optimizer, args.gamma, memory, n_reward_components)
                    update_target(policy_net, target_net, args.tau)

        if j >= args.exploration_episodes:
            print(f"Episode {i:6d} / {num_episodes:6d} ended, reward: {running_reward}")
        else:
            print(f"Pure exploration episode {j:6d} / {args.exploration_episodes:6d} done, reward: {running_reward}")
        if not args.evaluate:
            episode_rewards.append(running_reward)
            if ((i - start + 1) % args.checkpoint_interval) == 0 or i + 1 == num_episodes:
                torch.save(policy_net.state_dict(), f"{base_dir}/{subdir}/{i + 1}_policy.pt")
                print(f"Saved policy_net checkpoint:  {base_dir}/{subdir}/{i + 1}_policy.pt")
                torch.save(target_net.state_dict(), f"{base_dir}/{subdir}/{i + 1}_target.pt")
                print(f"Saved target_net checkpoint:  {base_dir}/{subdir}/{i + 1}_target.pt")
                torch.save(memory.state_dict(), f"{base_dir}/{subdir}/{i + 1}_memory.pt")
                print(f"Saved memory checkpoint:      {base_dir}/{subdir}/{i + 1}_memory.pt")
                with open(f"{base_dir}/{subdir}/{i + 1}_rewards", "wb") as file:
                    pickle.dump(episode_rewards, file)
                    print(f"Saved reward list checkpoint: {base_dir}/{subdir}/{i + 1}_rewards")
                with open(f"{base_dir}/{subdir}/{i + 1}_parameters", "wb") as file:
                    pickle.dump(args, file)
                    print(f"Saved argument checkpoint:    {base_dir}/{subdir}/{i + 1}_arguments")

                s = pd.Series(episode_rewards)
                sma = s.rolling(10).mean()
                fig, ax = plt.subplots()
                ax.plot(s, label = "Rewards (Raw)")
                ax.plot(sma, label = "Rewards (Moving Average)")
                ax.legend()
                plt.xlabel("Episode Number")
                plt.ylabel("Episode Reward")
                plt.title("RL Agent Reward Across Training Episodes")
                plt.savefig(f"{base_dir}/{subdir}/{i + 1}_plot.png")
                print(f"Saved reward plot checkpoint: {base_dir}/{subdir}/{i + 1}_plot.png")
        print("--------------------------------------------------------------------------------")

        if j >= args.exploration_episodes:
            i += 1
        j += 1

def update_target(policy_net, target_net, tau):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)

def optimize_model(policy_net, target_net, optimizer, gamma, memory, nrcmp):
    if len(memory) < memory._batch_size:
        return

    batch = memory.sample()
    states, actions, next_states, drewards, terminations = (batch.get(key) for key in ("state", "action", "next_state", "dreward", "terminated"))
    states = torch.cat([s for s in batch["state"]])
    actions = torch.cat([s for s in batch["action"]])
    next_states = torch.cat([s for s in batch["next_state"]])
    drewards = torch.cat([s for s in batch["dreward"]])
    terminations = torch.cat([s for s in batch["terminated"]])

    state_action_q_values = policy_net(states)
    state_action_values = torch.stack([
        state_action_q_values[_].gather(1, actions) for _ in range(state_action_q_values.shape[0])
    ]).transpose(0, 1).flatten(1, 2)
    with torch.no_grad():
        next_state_q_values = target_net(next_states)
        q_sum = next_state_q_values.transpose(0, 1).sum(dim = 1)
        next_state_values = torch.stack([
            next_state_q_values[_].gather(1, q_sum.max(1)[1].unsqueeze(1)) for _ in range(next_state_q_values.shape[0])
        ]).transpose(0, 1).flatten(1, 2)

    expected_state_action_values = drewards + (1 - terminations.unsqueeze(1).float()) * gamma * next_state_values

    td_errors = nn.functional.mse_loss(state_action_values, expected_state_action_values, reduction = "none")
    td_total = torch.sum(td_errors, dim = 1)
#    td_sums = torch.sum(td_errors, dim = 1)
#    for i in range(td_sums.shape[0]):
#    td_total = sum(td_errors)

    weights = batch.get("_weight")
    loss = (weights * td_total).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    batch.set("td_error", td_total)
    memory.update_tensordict_priority(batch)

def parse_arguments():
    parser = argparse.ArgumentParser(description = "Train and evaluate DQN agent for custom DogFight environment")

    parser.add_argument("--render", action = "store_true", help = "Draw the environment state to a window")
    parser.add_argument("--evaluate", action = "store_true", help = "Run the agent in evaluation mode")
    parser.add_argument("--checkpoint-dir", type = str, default = None, help = "Directory to load / save checkpoints")
    parser.add_argument("--load-checkpoint", type = int, default = None, help = "Checkpoint number to load")
    parser.add_argument("--checkpoint-interval", type = int, default = 100, help = "Episodic interval for checkpoint saving")
    parser.add_argument("--exploration-episodes", type = int, default = 250, help = "Number of pure exploration episodes")
    parser.add_argument("--num-episodes", type = int, default = 1000, help = "Number of training or evaluation episodes")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "Discount factor for future rewards")
    parser.add_argument("--eps-max", type = float, default = 1.00, help = "Epsilon start value")
    parser.add_argument("--eps-min", type = float, default = 0.03, help = "Epsilon end value")
    parser.add_argument("--memory-size", type = int, default = 1e6, help = "Size of replay buffer")
    parser.add_argument("--batch-size", type = int, default = 256, help = "Replay buffer sample batch size")
    parser.add_argument("--per-alpha", type = float, default = 0.65, help = "PER (prioritized experience replay) alpha value")
    parser.add_argument("--per-beta", type = float, default = 0.45, help = "PER (prioritized experience replay) beta value")
    parser.add_argument("--per-eps", type = float, default = 1e-6, help = "PER (prioritized experience replay) epsilon value")
    parser.add_argument("--tau", type = float, default = 0.005, help = "Soft-update rate between target and policy networks")
    parser.add_argument("--eval-eps", type = float, default = 0.0, help = "Epsilon value for evaluation")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    sys.exit(main())
