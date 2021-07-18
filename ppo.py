import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import gym
import numpy as np

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.next_is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.next_is_terminals[:]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim = -1)
        )
    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

class PPOAgent():
    def __init__(self, state_dim, action_dim, gamma, lr_actor, lr_crtic, epoch):
        self.gamma = gamma
        self.eps_clip = 0.2
        self.epoch = epoch
        self.actor = Actor(state_dim, action_dim)
        self.old_actor = Actor(state_dim, action_dim)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.critic = Critic(state_dim)
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.value_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_crtic)
        self.MseLoss = nn.MSELoss(reduction='mean')
        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state = torch.FloatTensor(state) #cast from ndarray to tensor
        action_probs = self.old_actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action).detach()
        return action, action_logprob
    
    def calc_rewards(self):
        rewards = []
        discounted_reward = self.critic(self.buffer.states[-1]).detach()
        for reward, next_is_terminal in zip(reversed(self.buffer.rewards[:-1]), reversed(self.buffer.next_is_terminals[:-1])):
            if next_is_terminal:
                discounted_reward = torch.zeros(1) # value of terminal state is 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        return torch.squeeze(torch.stack(rewards, dim=0), dim=0)
    
    # stateとaction→その確率と状態価値を返す
    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(actions)
        state_values = self.critic(states)
        return action_logprobs, state_values

    def update(self):

        ## n-step td error計算
        rewards = self.calc_rewards() #Tensor
        
        # advantage計算
        old_states = torch.squeeze(torch.stack(self.buffer.states[:-1], dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:-1], dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:-1], dim=0)).detach()

        for _ in range(self.epoch):
            action_logprobs, state_values = self.evaluate(old_states, old_actions)
            ratios = torch.exp(action_logprobs - old_logprobs)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2)
            value_loss = 0.5 * self.MseLoss(rewards, state_values)
            self.policy_optimizer.zero_grad()
            policy_loss.mean().backward()
            self.policy_optimizer.step()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        self.old_actor.load_state_dict(self.actor.state_dict())
        self.buffer.clear()

def train():
    env = gym.make("CartPole-v0")

    ## params ############
    state_dim = 4
    action_dim = 2
    episode = 1000
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
    epoch = 10
    update_time = 100
    #######################

    ppo_agent = PPOAgent(state_dim, action_dim, gamma, lr_actor, lr_critic, epoch)
    i_step = 0
    for i_episode in range(1, episode + 1):
        state = env.reset()
        done = False
        total_score = 0
        while not done:
            action, action_logprob = ppo_agent.select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            ppo_agent.buffer.states.append(torch.FloatTensor(state))
            ppo_agent.buffer.actions.append(action)
            ppo_agent.buffer.logprobs.append(action_logprob)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.next_is_terminals.append(done)

            i_step += 1
            if i_step % update_time == 0:
                ppo_agent.update()
            state = next_state
            total_score += reward
        if i_episode % 100 ==0:
            print(total_score)
        i_episode += 1

if __name__ == '__main__':
    train()