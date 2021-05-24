import numpy as np
import matplotlib.pyplot as plt
import gym
from functions import * 

def train(dim = "mid", alpha = 0.01, episode = 1000):
    env = gym.make("CartPole-v1")
    theta = np.random.random(feature(env.reset(), dim = dim).shape[0]*2)
    total_rewards = []
    for iter in range(episode):
        states = []
        actions = []
        rewards = []
        state = env.reset()
        states.append(state)
        total_reward = 0
        for step in range(500):
            action = get_action(theta, state, dim)
            next_state, reward, done, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            if done:
                break
            states.append(next_state)
            state = next_state
        
        total_reward_tmp = total_reward
        for i, (s, a) in enumerate(zip(states, actions)):
            g = total_reward_tmp
            theta = update(theta, s, a, g, alpha, dim)
            total_reward_tmp -= rewards[i]
        total_rewards.append(total_reward)
        if iter % 100 == 0:
            print(f"episode:{iter} reward:{total_reward}")
    fig = plt.figure()
    ax = fig.add_subplot(111, xlabel="episode", ylabel='total rewards')
    ax.plot(range(1,episode+1), total_rewards)
    plt.show()
    #return total_rewards # compare()を使うときは実行
    print(theta) # compare()を使うときは実行しない

if __name__ == "__main__":
    train()