import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
import os
import network
from network import PolicyNetwork, ValueNetwork

def train(episode=1000, learning_rate=1e-3):
    env = gym.make("CartPole-v1")
    policy_model = PolicyNetwork(env)
    value_model = ValueNetwork(env)
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr = learning_rate)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr = learning_rate)
    total_rewards = []
    for iter in range(episode):
        states = []
        state = env.reset()
        states.append(state)
        total_reward = 0

        policy_loss = torch.zeros(1, dtype = torch.float32)
        value_loss = torch.zeros(1, dtype = torch.float32)

        for step in range(500): # エピソード生成
            action_probs = policy_model(state).detach().numpy()
            action = np.random.choice([0, 1], p = action_probs)
            next_state, reward, done, _ = env.step(action)
            delta = reward + value_model(next_state) - value_model(state)
            value_loss += delta * delta
            policy_loss += -delta.detach() * torch.log(policy_model(state)[action])
            total_reward += reward
            if done:
                break
            states.append(next_state)
            state = next_state
        
        value_loss /= len(states)
        policy_loss /= len(states)
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        total_rewards.append(total_reward)
        if iter % 100 == 0:
            print(f"episode:{iter} reward:{total_reward}")
    return total_rewards

def draw(episode = 3000):
    total_rewards = train(episode=episode)
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel = "episode", ylabel='total rewards')
    ax.set_title("Actor Critic")
    ax.plot(range(1, episode + 1), total_rewards)
    save_dir = "./figure/"
    plt.savefig(os.path.join(save_dir, 'actor_critic.png'))

if __name__ == "__main__":
    draw()