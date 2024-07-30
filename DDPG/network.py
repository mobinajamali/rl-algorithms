import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class Critic(nn.Module):
    '''
    critic network to estimate the Q-value given a state-action pair
    '''
    def __init__(self, lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name, ckp__dir):
        super(Critic, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.ckp_dir = ckp__dir
        self.ckp_file = os.path.join(self.ckp_dir, name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # layer normalization
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # layer to handle the input of the action values 
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        # output layer for q-values
        self.q = nn.Linear(self.fc2_dims, 1)

        # weight initialization for fc1, fc2, output layer and action value layer
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_critic, weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        '''
        forward pass to compute the q-values
        '''
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_ckp(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_ckp(self):
        self.load_state_dict(T.load(self.ckp_file))


class Actor(nn.Module):
    '''
    actor network to determine the optimal action to take given a state
    '''
    def __init__(self, lr_actor, input_dims, fc1_dims, fc2_dims, n_actions, name, ckp__dir):
        super(Actor, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.ckp_dir = ckp__dir
        self.ckp_file = os.path.join(self.ckp_dir, name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # output layer for actions
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) 

        # initialization of layer weights and biases uniformely
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        # initialization for upper layer mu is +/- 0.003
        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr_actor)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x
    
    def save_ckp(self):
        T.save(self.state_dict(), self.ckp_file)

    def load_ckp(self):
        self.load_state_dict(T.load(self.ckp_file))

