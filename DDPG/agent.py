import torch as T
import torch.nn.functional as F
from network import Actor, Critic
from replay_buffer import Replay
from ou_noise import OUActionNoise
import numpy as np


class Agent():
    def __init__(self, lr_critic, lr_actor, input_dims, n_actions,
                 tau, gamma=0.99, mem_size=100000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):

        self.lr_critic = lr_critic
        self.lr_actor = lr_actor
        self.gamma = gamma
        self.tau = tau  # soft update parameter for updating target networks
        self.batch_size = batch_size  # batch size for sampling from the replay buffer

        # initialize replay buffer, noise process, and networks
        self.replay = Replay(mem_size, input_dims, n_actions) 
        self.noise = OUActionNoise(mu=np.zeros(n_actions))  
        self.actor = Actor(lr_actor, input_dims, fc1_dims, fc2_dims, n_actions, name='actor', ckp__dir='tmp')
        self.critic = Critic(lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name='critic', ckp__dir='tmp')
        self.target_actor = Actor(lr_actor, input_dims, fc1_dims, fc2_dims, n_actions, name='target_actor', ckp__dir='tmp')
        self.target_critic = Critic(lr_critic, input_dims, fc1_dims, fc2_dims, n_actions, name='target_critic', ckp__dir='tmp')

        # perform an update to initialize the target networks
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        '''
        choose an action based on the current state using the actor 
        network with added noise for exploration

        '''
        # set actor to the evaluation mode
        self.actor.eval()
        state = T.tensor(observation[np.newaxis, :], dtype=T.float, device=self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        # add noise for exploration
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # set actor back to training mode
        self.actor.train()

        return mu_prime.cpu().detach().numpy()[0]

    def save_model(self):
        self.actor.save_ckp()
        self.target_actor.save_ckp()
        self.critic.save_ckp()
        self.target_critic.save_ckp()

    def load_model(self):
        self.actor.load_ckp()
        self.target_actor.load_ckp()
        self.critic.load_ckp()
        self.target_critic.load_ckp()

    def store_transition(self, state, action, reward, state_, done):
        self.replay.store_transitions(state, action, reward, state_, done)

    def learn(self):
        '''
        perform a learning step by sampling from the replay buffer
        and updating the networks

        '''
        if self.replay.mem_ctr < self.batch_size:
            return 

        # sample a batch of transitions from the replay buffer and convert them to tensors
        state, action, reward, state_, done = self.replay.sample_buffer(self.batch_size)
        states = T.tensor(state, dtype=T.float).to(self.actor.device)
        actions = T.tensor(action, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(reward, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(done).to(self.actor.device)

        # forward pass through the target networks
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # update actor network
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # update target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        '''
        soft update the target networks using the tau value

        '''
        if tau is None:
            tau = self.tau

        # get parameters of the networks
        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # convert to dictionaries for easier access
        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        # soft update
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_state_dict[name].clone()

        # load the updated parameters into the target networks
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)