import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
         
class train_agent():

    def __init__(self, action_space, replace, epsilon, replay):
        self.current_network = None
        self.target_network = None
        self.memory = replay
        self.epsilon = epsilon
        self.action_space = action_space
        self.learn_step_ctr = 0
        self.replace = replace
        self.learning = None
        self.action = None

    def reset_epsilon(self, epsilon_start):
        self.epsilon.epsilon = epsilon_start
        self.epsilon.done = False

    def set_networks(self, current_network, target_network):
        self.current_network = current_network
        self.target_network = target_network

    def set_learning_strategy(self, strategy, learning_step):
        self.learning = strategy(gamma = 1, epsilon=self.epsilon, replace_target=self.replace, learning_step = learning_step)

    def set_action_selection(self, action):
         self.action = action

    def replace_network(self):
        if self.current_network != None and self.target_network != None:
                self.target_network.load_state_dict(self.current_network.state_dict())
        else:
            print("Missing Network! Hand network over wirh train_agent.set_networks")        

    def train(self, training_steps, environment, print_results, average):
        if self.current_network != None and self.target_network != None:
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
                        action = self.action.choose_action(observation, self.epsilon.epsilon)
                        #print(observation)
                        observation_, reward, done, _ = environment.step(action)
                        if print_results:
                            self.score += reward
                        self.memory.store_transition(observation, action, observation_, reward, done)

                        self.learning.learn(self.memory, self.current_network, self.target_network)    
                        observation = observation_
                    if print_results:
                        self.scores.append(self.score)
                        if len(self.scores) > average:
                            self.average_score.append(np.mean(self.scores[-50:]))
                    if((i%average==0) or i == training_steps-1 or i == 0):
                        print("average score:", self.average_score[-1:], " on training step ", i)
                                
                if print_results: 
                    #x = np.linspace(1, len(self.scores), len(self.scores))
                    x = np.linspace(1, len(self.average_score), len(self.average_score))
                    plt.figure(figsize=(30,15))
                    #plt.plot(x, self.scores, color = 'blue', linewidth =1, label = 'Overall_reward')
                    plt.plot(x, self.average_score, color = 'blue', linewidth =1, label = 'Average reward 50')

                    plt.xlabel('Games')
                    plt.ylabel('Reward')
                    plt.legend()
                    plt.grid(True)
                    #plt.savefig("Scores_512.pdf", format="pdf")
                    # %% 
                    plt.show()   
                    return self.average_score
        else:
            print("Missing Network! Hand network over wirh train_agent.set_networks")               

class action_selection_q_learning():
        
    def __init__(self, current_network, target_network, action_space ) -> None:
        self.current_network = current_network
        self.target_network = target_network
        self.action_space = action_space 

    def choose_action(self, observation, epsilon):
        
        if self.current_network != None and self.target_network != None:
                if np.random.random() > epsilon:
                    state = T.tensor([observation], dtype=T.float).to(self.current_network.device)
                    #actions, _ = self.current_network.forward(state)
                    actions = self.current_network.forward(state)
                    action = T.argmax(actions).item()
                else:
                    action = np.random.choice(self.action_space)
                return action    
        else:
            print("Missing Network! Hand network over wirh train_agent.set_networks")  
                     


class action_selection_dueling_q_learning():
        
    def __init__(self, current_network, target_network, action_space ) -> None:
        self.current_network = current_network
        self.target_network = target_network
        self.action_space = action_space 

    def choose_action(self, observation, epsilon):
        
        if self.current_network != None and self.target_network != None:
                if np.random.random() > epsilon:
                    state = T.tensor([observation], dtype=T.float).to(self.current_network.device)
                    A, V = self.current_network.forward(state)
                    actions = T.add(V, (A-T.mean(A, 1, False)), alpha=1)
                    action = T.argmax(actions).item()
                else:
                    action = np.random.choice(self.action_space)
                return action    
        else:
            print("Missing Network! Hand network over with train_agent.set_networks")          



























