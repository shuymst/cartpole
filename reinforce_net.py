import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
import network
from network import PolicyNetwork, ValueNetwork

def train(episode=1000, learning_rate=1e-3):
    env = gym.make("CartPole-v1")
    model = PolicyNetwork(env)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    total_rewards = []
    for iter in range(episode):
        states = []
        actions = []
        rewards = []
        state = env.reset()
        states.append(state)
        total_reward = 0
        for step in range(500):
            action_probs = model(state).detach().numpy()
            action = np.random.choice([0, 1], p = action_probs)
            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            if done:
                break
            states.append(next_state)
            state = next_state
        
        total_reward_tmp = total_reward
        loss = torch.zeros(1, dtype = torch.float32)
        for i, (s, a) in enumerate(zip(states, actions)):
            g = total_reward_tmp
            loss += -g * torch.log(model(s)[a]) # model(s)[a] = pi(a|s)
            total_reward_tmp -= rewards[i]
        #loss /= len(states)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_rewards.append(total_reward)
        if iter % 100 == 0:
            print(f"episode:{iter} reward:{total_reward}")
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel = "episode", ylabel='total rewards')
    ax.plot(range(1, episode + 1), total_rewards)
    plt.show()

if __name__ == "__main__":
    train()