import numpy as np

class Replay():
    def __init__(self, mem_size, input_dims, n_actions):
        self.mem_size = mem_size
        self.mem_ctr = 0

        self.state_mem = np.zeros((self.mem_size, *input_dims))
        self.new_state_mem = np.zeros((self.mem_size, *input_dims))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.done_mem = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, state_, done):
        '''
        Store a transition in the replay buffer
        '''
        # Calculate the index where the new transition should be stored 
        index = self.mem_ctr % self.mem_size
        self.state_mem[index] = state
        self.new_state_mem[index] = state_
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.done_mem[index] = done
        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        '''
        Sample a batch of transitions from the replay buffer
        '''
        # Calculate the number of transitions available for sampling
        num = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(num, batch_size, replace=False)
        state = self.state_mem[batch] 
        state_ = self.new_state_mem[batch] 
        action = self.action_mem[batch] 
        reward = self.reward_mem[batch] 
        done = self.done_mem[batch] 
        return state, action, reward, state_, done