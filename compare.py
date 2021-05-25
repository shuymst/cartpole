import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import os
from torch import nn
import network
from network import PolicyNetwork, ValueNetwork
import reinforce_net
import reinforce_baseline_net
import actor_critic

def main(episode = 1000):
    reinforce_history = []
    reinforce_baseline_history = []
    actor_critic_history = []
    for i in range(10):
        total_rewards = reinforce_net.train(episode = episode)
        reinforce_history.append(total_rewards)
    for i in range(10):
        total_rewards = reinforce_baseline_net.train(episode = episode)
        reinforce_baseline_history.append(total_rewards)
    for i in range(10):
        total_rewards = actor_critic.train(episode = episode)
        actor_critic_history.append(total_rewards)
    
    reinforce_average = np.mean(np.array(reinforce_history), axis = 0)
    reinforce_baseline_average = np.mean(np.array(reinforce_baseline_history), axis = 0)
    actor_critic_average = np.mean(np.array(actor_critic_history), axis = 0)

    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel = "episode", ylabel='total rewards')
    ax.set_title("10 times average")
    ax.plot(range(1, episode + 1), reinforce_average, color="blue", label="REINFORCE")
    ax.plot(range(1, episode + 1), reinforce_baseline_average, color="red", label="REINFORCE_Baseline")
    ax.plot(range(1, episode + 1), actor_critic_average, color="green", label="Actor Critic")
    ax.legend(loc=0)
    save_dir = "./figure/"
    plt.savefig(os.path.join(save_dir, 'compare.png'))

if __name__ == "__main__":
    main()
