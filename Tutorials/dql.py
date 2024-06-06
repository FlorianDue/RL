from replay_buffer import Replay_Buffer
from learning_and_exploration import epsilon_greedy, q_learning
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class deep_q_learning_agent():

    def __init__(self, current_network, target_network, max_size, batch_size, input_dims, action_space, replace, eps_start, eps_end, eps_dec):
        self.current_network = current_network
        self.target_network = target_network
        self.memory = Replay_Buffer(max_size, batch_size, input_dims)
        self.epsilon = epsilon_greedy(eps_start, eps_end, eps_dec)
        self.action_space = action_space
        self.learn_step_ctr = 0
        self.learning = q_learning(gamma = 1, epsilon=self.epsilon, replace_target=replace)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.current_network.device)
            actions = self.current_network.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action    
    
    def replace_network(self):
        self.target_network.load_state_dict(self.current_network.state_dict())

    def train(self, training_steps, environment, print_results, average):
        #set up new environment
        if print_results:
            self.scores = []
            self.average_score = []
        for i in range(training_steps):
            observation = environment.reset_env()
            if print_results:
                self.score = 0
                done = False
            while not done:
                action = self.choose_action(observation)
                observation_, reward, done = environment.step(action)
                if print_results:
                    self.score += reward
                self.memory.store_transition(observation, action, observation_, reward, done)

                self.learning.learn(self.memory, self.current_network, self.target_network)    
                observation = observation_
            if print_results:
                self.scores.append(self.score)
                if len(self.scores) > average:
                    self.average_score.append(np.mean(self.scores[-50:]))
            if((i%average==0) or i == training_steps-1):
                print("average score:", self.average_score[-1:])
                        
        if print_results: 
            #x = np.linspace(1, len(self.scores), len(self.scores))
            x = np.linspace(1, len(self.average_score), len(self.average_score))
            plt.figure(figsize=(30,15))
            #plt.plot(x, self.scores, color = 'blue', linewidth =1, label = 'Overall_reward')
            plt.plot(x, self.average_score, color = 'blue', linewidth =1, label = 'Average reward 50')

            plt.xlabel('Games')
            plt.ylabel('Reward')
            plt.legend()
            #plt.savefig("Scores_512.pdf", format="pdf")
            plt.show()             



        