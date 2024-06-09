import numpy as np
import torch as T

class Replay_Buffer():

    def __init__(self, max_size, batch_size, input_dims):
        #number of samples within the buffer
        self.mem_counter = 0
        self.max_mem_size = max_size
        self.batch_size = batch_size
        self.replace = False
        #create list for the stored elements to be sampled
        #current state
        self.state_memory = np.zeros((self.max_mem_size, input_dims), dtype = np.float32)
        #next_state
        self.new_state_memory = np.zeros((self.max_mem_size, input_dims), dtype = np.float32)
        #taken actions
        self.action_memory = np.zeros(self.max_mem_size, dtype = np.float32)
        #received reward
        self.reward_memory = np.zeros(self.max_mem_size, dtype = np.float32)
        #store information if the sample was the last sample of an episode
        self.terminal_memory = np.zeros(self.max_mem_size, dtype = np.float32)

    def store_transition(self, state, action, new_state, reward, done):
        self.state_memory[self.mem_counter] = state
        self.new_state_memory[self.mem_counter] = new_state
        self.reward_memory[self.mem_counter] = reward
        self.action_memory[self.mem_counter] = action
        self.terminal_memory[self.mem_counter] = done
        #if the max size of the buffer is reached,\
        # new samples will replace the oldest samples
        if(self.max_mem_size == self.mem_counter):
            self.mem_counter = 0
            self.replace = True
        else:    
            self.mem_counter +=1

    def sample_data(self):
        max_mem = self.max_mem_size if self.replace == True else self.mem_counter
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        return  T.tensor(self.state_memory[batch]).to(T.long), \
                T.tensor(self.action_memory[batch]).to(T.long),\
                T.tensor(self.reward_memory[batch]).to(T.long),\
                T.tensor(self.new_state_memory[batch]).to(T.long),\
                T.tensor(self.terminal_memory[batch]).to(T.long)

    def reset_memory(self, input_dims, batch_size):
        self.state_memory = np.zeros((self.max_mem_size, input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.max_mem_size, input_dims), dtype = np.float32)
        self.action_memory = np.zeros(self.max_mem_size, dtype = np.float32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype = np.float32)  
        self.replace = False
        self.mem_counter = 0      
        self.batch_size = batch_size

