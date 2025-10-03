import time

from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

def make_env(gym_id, seed, idx):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = RGBImgPartialObsWrapper(env)  # replace "obs" with partial RGB image
        env = ImgObsWrapper(env)           # turn dict obs -> np.array
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder="grid_vid",
                episode_trigger=lambda t: t % 100 == 0
            )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
     torch.nn.init.orthogonal_(layer.weight, std)
     torch.nn.init.constant_(layer.bias, bias_const)
     return layer

class vpg_Agent(nn.Module):
    def __init__(self, env):
        super(vpg_Agent, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride = 1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(576, 128)),
            nn.ReLU()
        )
        
        self.actor = layer_init(nn.Linear(128, env.action_space.n), std=0.1)
        
    def get_action_and_probs(self, x, action=None):
        x = x.permute(0, 3, 1, 2)
        features = self.network(x/255)
        logits = self.actor(features)
        probs = Categorical(logits=logits) #create a probability dist from the logits
        if action is None: #this is in the rollout phase
            action = probs.sample() #sample a random action according to the distribution 
        return action, probs.log_prob(action), probs.entropy()


if __name__ == "__main__":
    run_name = f"Unlock_{int(time.time())}"
    writer = SummaryWriter(f"vpg_runs/{run_name}")

    env = gym.make("MiniGrid-Empty-6x6-v0", render_mode="rgb_array")
    env = RGBImgPartialObsWrapper(env)  # replace "obs" with partial RGB image
    env = ImgObsWrapper(env)           # turn dict obs -> np.array
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.RecordVideo(env, video_folder="vpg_vid", episode_trigger=lambda t: t % 100 == 0)

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)    
    env.action_space.seed(1)
    env.observation_space.seed(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs, _ = env.reset()
    #obs is of shape 56,56,3 (rgb 56x56 image)
    obs = torch.tensor(obs, dtype=torch.float32)
    obs = obs.unsqueeze(0)  # adds batch dimension at index 0

    print(env.action_space.n)
    agent = vpg_Agent(env).to(device)

    #action, logprobs, entropy = agent.get_action_and_probs(obs)
    #print(action.item())
    #env.step(action.cpu().numpy())

    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps = 1e-5)
    gamma = 0.99
    ent_coeff = 0.01
    num_updates = 10000
    global_step = 0

    #actions, rewards, obs
    
    next_obs, _ = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
    next_obs = next_obs.unsqueeze(0)
    for update in range(1, num_updates+1):
        done = False
        #rollout one episode
        states, actions, rewards, logprobs, entropys = [], [], [], [], []
        while(not done):
            global_step+=1
            states.append(next_obs)
            action, logprob, entropy = agent.get_action_and_probs(next_obs)
            actions.append(action)
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            next_obs = next_obs.unsqueeze(0)
            rewards.append(reward)
            entropys.append(entropy)
            logprobs.append(logprob)
            done = terminated or truncated
        
        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        writer.add_scalar("charts/episodic_return", info['episode']['r'], global_step)
        writer.add_scalar("charts/episodic_length", info['episode']['l'], global_step)

        next_obs, _ = env.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        next_obs = next_obs.unsqueeze(0)

        length = len(rewards)
        states = torch.cat(states).to(device)            # shape [T, obs_dim]
        actions = torch.stack(actions).to(device)        # shape [T]
        rewards = torch.tensor(rewards).to(device)       # shape [T]
        logprobs = torch.stack(logprobs).to(device)                 # [T]
        entropys = torch.stack(entropys).to(device)

        #compute returns
        returns = torch.zeros_like(rewards, dtype=torch.float32).to(device)
        for t in reversed(range(length)):
            if t == length-1:
                returns[t] = rewards[t]
            else:
                returns[t] = rewards[t] + gamma*returns[t+1]

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        #compute loss
        policy_loss = -(returns*logprobs).mean()
        entropy_loss = entropys.mean()

        loss = policy_loss - ent_coeff * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)

    env.close()