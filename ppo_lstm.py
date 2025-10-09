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
                video_folder="ppo_lstm_vid",
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
            layer_init(nn.Linear(576, 512)),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(512, 512)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512,1), std=1)

    def get_states(self, x, lstm_state, done):
        hidden = self.network(x / 255.0)
        #LSTM expects inputs in shape (seq_len, batch, input_size)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            #reset the memory state on Done=True
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        x = x.permute(0, 3, 1, 2)
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)
    
    def get_action_and_value(self, x, lstm_state, done, action=None):
        x = x.permute(0, 3, 1, 2)
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden) #unnormalized log action probabilities
        probs = Categorical(logits=logits) #create a probability dist from the logits
        if action is None: #this is in the rollout phase
            action = probs.sample() #sample a random action according to the distribution 

        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state
    
    def features(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.network(x)
    
if __name__ == "__main__":
    run_name = f"run_{int(time.time())}"
    writer = SummaryWriter(f"ppo_lstm_runs/{run_name}")

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = 8
    envs = gym.vector.SyncVectorEnv([make_env("MiniGrid-MemoryS7-v0", 1+ i, i) for i in range(num_envs)])

    observation, _ = envs.reset()
    observation = torch.from_numpy(observation).float().to(device)
    agent = Agent(envs).to(device)
    lr = 3e-4
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr, eps = 1e-5)
    

    print(observation.shape)
    print(agent.features(observation).shape)


    num_steps = 128
    gamma = 0.99
    num_updates = 100000
    ent_coef = 0.025
    val_coef = 0.5
    lmbda = 0.95
    clip = 0.2
    epochs = 4
    batch_size = (num_envs*num_steps)
    minibatch_size = batch_size//4
    minibatches = batch_size//minibatch_size


    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    print("obs.shape", obs.shape)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    print("actions.shape", actions.shape)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    observation, _ = envs.reset()
    next_obs = torch.tensor(observation, dtype=torch.float32).to(device)
    next_done = torch.zeros(num_envs).to(device)
    global_step = 0

    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
    )

    for update in range(1, num_updates+1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * lr
        optimizer.param_groups[0]["lr"] = lrnow

        for t in range(num_steps):
            global_step+=1
            obs[t] = next_obs
            dones[t] = next_done
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
            actions[t] = action
            logprobs[t] = logprob
            values[t] = value.flatten()
            
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).to(device)
            rewards[t] = reward
            done = terminated | truncated
            next_done = torch.tensor(done, dtype=torch.float32).to(device)
            if "episode" in info:
                for env_index in range(num_envs):
                    if info["_episode"][env_index]:
                        print(f"[env {env_index}] global_step={global_step}, episodic_return={info['episode']['r'][env_index]}")
                        writer.add_scalar("charts/episodic_return", info['episode']['r'][env_index], global_step)
                        writer.add_scalar("charts/episodic_length", info['episode']['l'][env_index], global_step)
                        break

        #GAE compute returns and advantages
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            next_value = agent.get_value(next_obs, next_lstm_state, next_done).flatten()
            advantages = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == (num_steps-1):
                    nextnonterminal = 1 - next_done
                    advantages[t] = rewards[t]+ gamma*next_value*nextnonterminal - values[t]
                else:
                    nextnonterminal = 1 - dones[t+1]
                    delta = rewards[t] + gamma*values[t+1]*nextnonterminal - values[t]
                    advantages[t] = delta + gamma*lmbda*nextnonterminal*advantages[t+1]
            returns = advantages + values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_logprobs = logprobs.reshape(-1)
        b_values = values.reshape(-1)


        envsperbatch = num_envs // minibatches
        b_inds = np.arange(batch_size)
        envinds = np.arange(num_envs)
        flatinds = np.arange(batch_size).reshape(num_steps, num_envs)
        for epoch in range(epochs):
            np.random.shuffle(envinds)
            for start in range(0, num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, mb_newlogprobs, mb_entropies, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.squeeze()[mb_inds],
                )

                
                mb_returns = b_returns[mb_inds]
                mb_advantages = b_advantages[mb_inds]
                mb_values = b_values[mb_inds]

                logratio = mb_newlogprobs - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio-1) - logratio).mean()


                #calculate policy loss
                pg_loss1 = -mb_advantages*ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip, 1+clip)
                #max of negatives, paper does min of negatives
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                #calculate value loss
                newvalue = newvalue.flatten()
                v_loss_unclipped = (newvalue - mb_returns) **2
                v_clipped = mb_values + torch.clamp(newvalue - mb_values, -clip, clip)
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) **2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                #compute total loss
                entropy_loss = mb_entropies.mean()
                loss = policy_loss - ent_coef * entropy_loss + v_loss*val_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            #logging
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("charts/kl", approx_kl.item(), global_step)

    envs.close()