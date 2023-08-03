import argparse
import copy
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import torch

import gym_env

from datetime import datetime
from itertools import count
from tensordict import TensorDict
from time import sleep
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

    msep_explanations = [
        "I will preserve ammunition and miss fewer bullets",
        "I am more likely to successfully shoot the enemy",
        "I am more likely to successfully shoot the target",
        "I will avoid colliding with the enemy and its missile",
        "I need to stay within the combat zone and complete my mission",
        "I need to complete the mission in a timely manner",
        "I need to approach the target's firing zone to land a successful hit"
    ]
    msen_explanations = [
        "I will waste more ammunition",
        "I am less likely to successfully hit the enemy",
        "I am less likely to successfully hit the target",
        "I would most likely collide with the enemy and its missile",
        "I would be leaving the combat zone and abandon the mission",
        "I will waste time",
        "I am increasing my distance from the target such that I can't successfully shoot it"
    ]

    # Set the random seed
    seed = args.seed
    if seed is None:
        seed = int(datetime.now().timestamp())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    env = gym.make("gym_env/DogFight", time_limit = args.episode_time, eval_mode = args.evaluate, render_mode = render_mode)

    n_actions = env.action_space.n
    state, info = env.reset()
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
        rewind_memory = []
        state, info = env.reset()
        start_state = env.unwrapped.get_state()
        paused_state = start_state
        state = torch.tensor(state, device = device, dtype = torch.float32).unsqueeze(0)
        rewind_memory.append((start_state, copy.deepcopy(state)))
        running_reward = 0
        eps = 0 if args.evaluate else max(args.eps_min, eps_max * eps_decay ** i)

        print("--------------------------------------------------------------------------------")
        if j >= args.exploration_episodes:
            print(f"Episode {i:6d} / {num_episodes:6d} started, epsilon: {eps}")
        else:
            print(f"Pure exploration episode {j:6d} / {args.exploration_episodes:6d} started")

        shooting_transitions = []
        shooting_flags = []
#        last_paused = False
        # (True) factual index
        tfact_idx = 0
        tfact_done = False
        # Counter factual index
        cfact_idx = 0
        cfact_done = False
        # Step limit for performing factual / counterfactual trajectories
        traj_limit = 0
#        paused_state = None
        # Store the states for the factual and counterfactual scenarios
        # to do batch processing for the Q-values afterwards
        tfact_states = torch.tensor([])
        cfact_states = torch.tensor([])
        qsa1 = torch.zeros(n_reward_components, 1)
        qsa2 = torch.zeros(n_reward_components, 1)
        do_split_facts = False
        is_paused = False
        for t in count():
            if env.unwrapped.rewind:
#                paused_state = rewind_memory.pop()
#                env.unwrapped.set_state(paused_state)
                if len(rewind_memory) > 1:
                    tmp_state, state = rewind_memory.pop()
                    env.unwrapped.set_state(tmp_state)
                    paused_state = tmp_state
                else:
                    env.unwrapped.set_state(rewind_memory[0][0])
                    paused_state = rewind_memory[0][0]
                    state = rewind_memory[0][1]

            if not is_paused and env.unwrapped.paused:
                is_paused = True
                paused_state = rewind_memory[-1][0]
                state = rewind_memory[-1][1]
#                env_state = env.unwrapped.get_state()
#                paused_state = env.unwrapped.get_state()
#                paused_state = env_state
#                rewind_memory.append(env_state)
            elif is_paused and not env.unwrapped.paused:
                is_paused = False
                env.unwrapped.draw_actions = False
#                paused_state = None

            if not do_split_facts and env.unwrapped.perform_split_facts:
                do_split_facts = True
            elif do_split_facts and not env.unwrapped.perform_split_facts:
                do_split_facts = False

#            if env.unwrapped.perform_split_facts:
            if do_split_facts:
                if (not tfact_done and tfact_idx == 0) and (not cfact_done and cfact_idx == 0):
#                    paused_state = env.unwrapped.get_state()
                    traj_limit = len(env.unwrapped.counterfactual_trajectories)
                    env.unwrapped.draw_actions = True

#                if tfact_idx < traj_limit:
                if not tfact_done:
                    with torch.no_grad():
#                        out = policy_net(state)
#                        action = out.sum(dim = 0).max(dim = 1)[1].view(1, 1)
#                        qsa1 += out[:, :, action.item()]

                        # If agent network chooses to shoot enemy when not visible,
                        # force it to choose next highest q-val action
                        out = policy_net(state)
                        cand_actions = out.sum(dim = 0).topk(k = 2, dim = 1)
#                        cand_actions = sum(policy_net(state)).topk(k = 2, dim = 1)
                        if (
                            cand_actions[1][0][0].item() == env.unwrapped.ACTION_SHOOT_ENEMY and
                            state[0][4].item() == 0
                        ):
                            action = cand_actions[1][0][1].view(1, 1)
                            qsa1 += out[:, :, cand_actions[1][0][1].item()]
                        else:
                            action = cand_actions[1][0][0].view(1, 1)
                            qsa1 += out[:, :, cand_actions[1][0][0].item()]
                    tfact_idx += 1
#                elif cfact_idx < traj_limit:
                elif not cfact_done:
                    if cfact_idx == 0:
                        env.unwrapped.set_state(paused_state)
                    action = torch.tensor(
                        [[env.unwrapped.counterfactual_trajectories[cfact_idx]]],
                        device = device,
                        dtype = torch.long
                    )
                    cfact_idx += 1
                else:
                    print(f"Got here @:\ntfact_idx = {tfact_idx}\ncfact_idx = {cfact_idx}")
                    cfact_actions = torch.tensor([env.unwrapped.counterfactual_trajectories]).T
                    cfact_out = policy_net(cfact_states)
#                    cfact_qvals = torch.stack([cfact_out[_].gather(1, cfact_actions) for _ in range(cfact_out.shape[0])]).transpose(0, 1)
#                    cfact_qvals = torch.stack([cfact_out[_].gather(1, cfact_actions[:cfact_idx + 1]) for _ in range(cfact_out.shape[0])]).transpose(0, 1)
                    cfact_qvals = torch.stack([cfact_out[_].gather(1, cfact_actions[:cfact_idx]) for _ in range(cfact_out.shape[0])]).transpose(0, 1)
                    qsa2 = torch.sum(cfact_qvals, dim = 0)
#                    print(f"qsa1:\n{qsa1}")
#                    print(f"qsa2:\n{qsa2}")
                    dc = qsa1 - qsa2
#                    print(f"dc:\n{dc}")
                    mask = (qsa2 > qsa1).float()
#                    print(f"mask:\n{mask}")
                    disadvantage = (mask * dc.abs()).sum(dim = 0)
#                    print(f"disadvantage:\n{disadvantage.item()}")
                    tmp_p = dc.sort(dim = 0, descending = True)
                    order_dcp = dc.sort(dim = 0, descending = True)[0]
#                    print(f"descending dc:\n{order_dcp}")
                    msep = torch.tensor([[order_dcp[:_].sum(dim = 0) > disadvantage for _ in range(1, order_dcp.shape[0] + 1)]]).T
#                    print(f"mse+:\n{msep}")
                    msep_ind = msep.float().argmax()
#                    print(f"mse+ argmax: {msep_ind.item()}")
#                    print(f"corresponding index in original dc list: {tmp_p[1][msep_ind].item()}")
                    print(f"\n\nAdvantage to my trajectory:")
                    for _ in range(msep_ind.item() + 1):
                        print(f"{msep_explanations[tmp_p[1][_].item()]}")
                    tmp_n = dc.sort(dim = 0, descending = False)
                    print(f"\n\nDisadvantages to my trajectory:")
                    for _ in range(msep_ind.item() + 1):
                        if tmp_n[0][_] < 0:
                            print(f"{msen_explanations[tmp_n[1][_].item()]}")


                    cfact_states = torch.tensor([])
                    tfact_idx = 0
                    tfact_done = False
                    cfact_idx = 0
                    cfact_done = False
                    traj_limit = 0
                    do_split_facts = False
                    env.unwrapped.perform_split_facts = False
                    env.unwrapped.set_state(paused_state)
                    state = rewind_memory[-1][1]
#                    is_paused = True
#                    env.unwrapped.paused = True

#            if not env.unwrapped.perform_split_facts:
            if not do_split_facts:
                if args.evaluate or random.random() > eps:
                    with torch.no_grad():
#                        action = sum(policy_net(state)).max(dim = 1)[1].view(1, 1)
                        # If agent network chooses to shoot enemy when not visible,
                        # force it to choose next highest q-val action
                        cand_actions = sum(policy_net(state)).topk(k = 2, dim = 1)
                        if (
                            cand_actions[1][0][0].item() == env.unwrapped.ACTION_SHOOT_ENEMY and
                            state[0][4].item() == 0
                        ):
                            action = cand_actions[1][0][1].view(1, 1)
                        else:
                            action = cand_actions[1][0][0].view(1, 1)

                else:
                    if state[0][4].item() == 0:
                        action = torch.tensor([[env.unwrapped.ACTION_SHOOT_ENEMY]])
                        while action.item() == env.unwrapped.ACTION_SHOOT_ENEMY:
                            action = torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)
                    else:
                        action = torch.tensor([[env.action_space.sample()]], device = device, dtype = torch.long)

            next_state, reward, terminated, truncated, info = env.step(action.item())
            if is_paused and not do_split_facts:
                continue
#            if env.unwrapped.paused or (not env.unwrapped.paused and last_paused):
#                continue
#            if env.unwrapped.paused:
#                paused_state = env.unwrapped.get_state()

#            last_unpaused = not env.unwrapped.get_wrapper_attr("paused")
#            if not env.unwrapped.perform_split_facts:
#            if not do_split_facts:
            if not is_paused:
                running_reward += reward
            dreward = info["dreward"]
            dreward = torch.tensor([dreward], device = device)
            next_state = torch.tensor(next_state, device = device, dtype = torch.float32).unsqueeze(0)
            done = terminated or truncated
            terminated = torch.tensor([terminated], device = device, dtype = torch.bool)

#            if env.unwrapped.perform_split_facts:
            if do_split_facts:
                if tfact_done and not cfact_done:
#                if cfact_idx < traj_limit and tfact_idx >= traj_limit:
                    cfact_states = torch.cat((cfact_states, next_state), 0)

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

#            if not env.unwrapped.perform_split_facts and info["shoot_act"]:
            if not do_split_facts and info["shoot_act"]:
                shooting_transitions.append(transition)
                shooting_flags.append(info)
            elif not args.evaluate:
                memory.add(transition)
                if j >= args.exploration_episodes:
                    optimize_model(policy_net, target_net, optimizer, args.gamma, memory, n_reward_components)
                    update_target(policy_net, target_net, args.tau)

            if done:
                if not env.unwrapped.perform_split_facts:
                    break
                if not tfact_done:
                    tfact_done = True
                elif not cfact_done:
                    cfact_done = True
#                if tfact_idx < traj_limit:
#                    tfact_idx = traj_limit + 1
#                elif cfact_idx < traj_limit:
#                    cfact_idx = traj_limit + 1

#            if env.unwrapped.perform_split_facts:
            if do_split_facts:
                if not tfact_done and tfact_idx >= traj_limit:
                    tfact_done = True
                elif not cfact_done and cfact_idx >= traj_limit:
                    cfact_done = True

            if not is_paused:
                rewind_memory.append((env.unwrapped.get_state(), copy.deepcopy(next_state)))

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
#        q_sum = next_state_q_values.transpose(0, 1).sum(dim = 1)
#        next_state_values = torch.stack([
#            next_state_q_values[_].gather(1, q_sum.max(1)[1].unsqueeze(1)) for _ in range(next_state_q_values.shape[0])
#        ]).transpose(0, 1).flatten(1, 2)
        qt_amax = next_state_q_values.transpose(0, 1).sum(dim = 1).max(1)[1].unsqueeze(1)
        next_state_values = torch.stack([
            state_action_q_values[_].gather(1, qt_amax) for _ in range(state_action_q_values.shape[0])
        ]).transpose(0, 1).flatten(1, 2)
#        next_state_values = torch.stack([
#            next_state_q_values[_].gather(1, q_sum.max(1)[1].unsqueeze(1)) for _ in range(next_state_q_values.shape[0])
#        ]).transpose(0, 1).flatten(1, 2)

    expected_state_action_values = drewards + (1 - terminations.unsqueeze(1).float()) * gamma * next_state_values

    td_errors = nn.functional.mse_loss(state_action_values, expected_state_action_values, reduction = "none")
    td_total = torch.sum(td_errors, dim = 1)

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
    parser.add_argument("--tau", type = float, default = 0.005, help = "Soft-update factor between target and policy networks")
    parser.add_argument("--seed", type = int, default = None, help = "Random seed to use")
    parser.add_argument("--episode-time", type = int, default = 60, help = "Episode time limit (in seconds)")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    sys.exit(main())
