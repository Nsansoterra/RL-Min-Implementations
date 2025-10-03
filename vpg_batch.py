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
                video_folder="vpgb_vid",
                episode_trigger=lambda t: t % 50 == 0
            )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
     torch.nn.init.orthogonal_(layer.weight, std)
     torch.nn.init.constant_(layer.bias, bias_const)
     return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()

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

        self.actor = layer_init(nn.Linear(128, envs.single_action_space.n), std=0.01)
    
    def get_action_and_probs(self, x, action=None):
        x = x.permute(0, 3, 1, 2)
        hidden = self.network(x/255.0)
        logits = self.actor(hidden) #unnormalized log action probabilities
        probs = Categorical(logits=logits) #create a probability dist from the logits
        if action is None: #this is in the rollout phase
            action = probs.sample() #sample a random action according to the distribution 

        return action, probs.log_prob(action), probs.entropy()
    
    def features(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.network(x)
    


if __name__ == "__main__":
    run_name = f"Unlock_{int(time.time())}"
    writer = SummaryWriter(f"vpgb_runs/{run_name}")

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = 8
    envs = gym.vector.SyncVectorEnv([make_env("MiniGrid-Empty-8x8-v0", 1+ i, i) for i in range(num_envs)])
    
    observation, _ = envs.reset()
    observation = torch.from_numpy(observation).float().to(device)
    agent = Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps = 1e-5)

    print(observation.shape)
    print(agent.features(observation).shape)
    num_steps = 128
    gamma = 0.99
    num_updates = 100000
    ent_coef = 0.01
    batch_size = (num_envs*num_steps)
    minibatch_size = batch_size//4
    epochs = 4
    #states (obs), actions, rewards, dones
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    print("obs.shape", obs.shape)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    print("actions.shape", actions.shape)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    
    observation, _ = envs.reset()
    next_obs = torch.tensor(observation, dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)
    global_step = 0

    for update in range(1, num_updates + 1):
        #collect trajectories
        for t in range(num_steps):
            global_step+=1 
            dones[t] = next_done
            obs[t] = next_obs
            with torch.no_grad():
                action, logprob, _ = agent.get_action_and_probs(next_obs)
                actions[t] = action
                logprobs[t] = logprob


            next_obs, reward, truncated, terminated, info = envs.step(action.cpu().numpy())
            done = truncated | terminated
            rewards[t] = torch.tensor(reward, dtype=torch.float32).to(device)
            next_done = torch.tensor(done, dtype=torch.float32).to(device)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)

            if "episode" in info:
                for env_index in range(num_envs):
                    if info["_episode"][env_index]:
                        print(f"[env {env_index}] global_step={global_step}, episodic_return={info['episode']['r'][env_index]}")
                        writer.add_scalar("charts/episodic_return", info['episode']['r'][env_index], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode']['l'][env_index], global_step)
                        break

        #compute discounted return
        
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            for t in reversed(range(num_steps)):
                if t == num_steps-1:
                    nextnonterminal = 1.0 - next_done
                    returns[t] = rewards[t]
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    returns[t] = rewards[t] + nextnonterminal*gamma*returns[t+1]
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        _, newlogprob, entropy = agent.get_action_and_probs(
            b_obs, b_actions.long()
        )

        # compute losses
        policy_loss = -(newlogprob * b_returns).mean()
        entropy_loss = entropy.mean()

        loss = policy_loss - ent_coef * entropy_loss

        # single update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)



    envs.close()
    