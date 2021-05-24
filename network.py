import torch
from torch import nn

class PolicyNetwork(nn.Module):
    def __init__(self, env):
        super(PolicyNetwork, self).__init__()
        self.input_num = env.observation_space.shape[0]
        self.output_num = env.action_space.n
        self.layers = nn.Sequential(
            nn.Linear(self.input_num, 20),
            nn.ReLU(),
            nn.Linear(20, self.output_num),
            nn.Softmax(dim = -1)
        )
    
    def forward(self, state):
        x = torch.from_numpy(state).float()
        action_probs = self.layers(x)
        return action_probs

class ValueNetwork(nn.Module):
    def __init__(self, env):
        super(ValueNetwork, self).__init__()
        self.input_num = env.observation_space.shape[0]
        self.output_num = 1
        self.layers = nn.Sequential(
            nn.Linear(self.input_num, 20),
            nn.ReLU(),
            nn.Linear(20, self.output_num)
        )
    
    def forward(self, state):
        x = torch.from_numpy(state).float()
        return self.layers(x)