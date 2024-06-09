import gym
import numpy as np
import random

class OAAP(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, episode_length, production_lines, nbr_resources_per_line, resource_capacity, min_rev, max_rev, order_set, action_space, penalty_reward):
        
        super(OAAP, self)
        
        self.episode_length = episode_length
        self.production_lines = production_lines
        self.nbr_resources_per_line = nbr_resources_per_line
        self.resource_capacity = resource_capacity
        self.min_rev = min_rev
        self.max_rev = max_rev
        self.action_space = action_space
        self.penalty_reward = penalty_reward
        self.current_step = None

        if order_set:
            self.order_set = order_set
        else:
            #default oder set for 6 resoruces
            self.order_set = np.array(
                [[1, 2, 3, 2, 1, 0],
                [1, 1, 1, 1, 0, 0],
                [0, 2, 2, 0, 0, 0],
                [2, 2, 1, 1, 0, 0],
                [1, 3, 2, 2, 1, 0],
                [2, 1, 1, 1, 3, 3]])

        #set up the shop floor
        self.resources = np.zeros((self.production_lines, self.nbr_resources_per_line), dtype = int)
        #set up the capacity
        self.capacity = self.resource_capacity
        #if isinstance(self.resource_capacity, list):
        #    #use pre-set capacity for resources, production lines
        #    self.capacity = self.resource_capacity
        #else:
        #    #set single value for all resources
        #    self.capacity = np.zeros((self.production_lines, self.resource_capacity), dtype = int)
            


    def print_reset(self):
        print(self.resources)
        print("self.resources")
        print(self.capacity)
        print("self.capacity")
        print(self.request_type)
        print("self.request_type")
        print(self.request_reward_per_capa)
        print("self.request_reward_per_capa")
        print(self.total_reward_per_requ)
        print("self.total_reward_per_requ")
        print(self.requests)
        print("self.requests")

    def reset_env(self):           
        self.current_step = 0
        self.recent_reward = 0
        self.total_reward = 0
        self.total_lost_reward = 0
        self.accepted = 0
        self.declined = 0
        self.done = False
        #if isinstance(self.resource_capacity, list):
        #    self.state_capacity = self.resource_capacity
        #else:
        self.state_capacity = np.ones((self.production_lines, self.resource_capacity), dtype = int) * self.capacity
        self.requests = np.zeros((self.episode_length, self.resource_capacity), dtype = int)
        self.request_type = np.random.randint(0, len(self.order_set), self.episode_length, int)
        self.request_reward_per_capa = np.random.randint(self.min_rev, self.max_rev+1, self.episode_length, int)
        self.total_reward_per_requ = np.zeros(self.episode_length, int)
        #currently not possible to resditribute the required capacity between resources
        for i in range(0, self.episode_length):
            self.requests[i] = self.order_set[self.request_type[i]]
            req_capa = 0
            for j in range(len(self.requests[i])):
                req_capa += self.requests[i][j]   
            self.total_reward_per_requ[i] = req_capa * self.request_reward_per_capa[i]
        return self.observe()    

    def observe(self):
        self.observation = np.append(self.requests[self.current_step], self.state_capacity)
        remaining_steps = self.episode_length - self.current_step
        self.observation = np.append(self.observation, [remaining_steps, self.total_reward_per_requ[self.current_step]])
        return self.observation

    def observe_final_step(self):
        self.observation = np.append(np.zeros((self.production_lines * self.nbr_resources_per_line), dtype = int), self.state_capacity)
        remaining_steps = self.episode_length - self.current_step
        self.observation = np.append(self.observation, [remaining_steps, 0])
        return self.observation
    
    def step(self, action):
        #if the action is the last in the action action space, the order is decline
        if(action == len(self.action_space)-1 ):
            self.recent_reward = 0
            self.declined +=1
            self.total_lost_reward += self.total_reward_per_requ[self.current_step]
        else:
            self.accepted +=1
            self.state_capacity = self.state_capacity - self.requests[self.current_step]
            #compute reward, penalty_reward in case that the capacity is exceeded
            if(np.min(self.state_capacity) < 0):
                self.recent_reward = -self.penalty_reward
                self.total_reward -= self.penalty_reward
                for i in range(len(self.state_capacity[0])):
                    if(self.state_capacity[0][i] < 0):
                        self.state_capacity[0][i] = 0
            else:
                self.recent_reward = self.total_reward_per_requ[self.current_step]
                self.total_reward += self.total_reward_per_requ[self.current_step]
        self.current_step += 1
        if(self.current_step > self.episode_length - 1):
            self.done = True
            self.observe_final_step()
        else:
            self.observe()
        return self.observation, self.recent_reward, self.done        
