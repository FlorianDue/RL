# some_file.py
#import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, '../Environments')
import sys
sys.path.append('Agents')

from Neural_Networks.nn import dqn_neural_network, dueling_neural_network
from Agents.agent import train_agent, action_selection_q_learning, action_selection_dueling_q_learning
from Environments.Order_Acceptance_and_Allocation import OAAP
from Agents.utils.learning_and_exploration import epsilon_greedy
from Agents.utils.replay_buffer import Replay_Buffer
from Agents.utils.learning_and_exploration import double_q_learning, learning, q_learning, dueling_q_learning, dueling_double_q_learning
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


# episode_length, production_lines, nbr_resources_per_line, resource_capacity, min_rev, max_rev, order_set, action_space,
env = OAAP(episode_length = 20,
    production_lines = 1,
    nbr_resources_per_line = 6,
    resource_capacity = 6,
    min_rev = 1,
    max_rev = 9,
    action_space = [0, 1],
    order_set = None,
    penalty_reward = 20)


def set_up_dueling_dqn_networks():
    #path, input_dims, fc1_dims ,fc2_dims, n_actions, loss, activation_function, lr, dropout
    current_network = dueling_neural_network(path = '.test.zip', input_dims=14, fc1_dims=256,  fc2_dims=256, n_actions= 2,
                         loss = nn.MSELoss(), activation_function= [F.relu, F.relu], lr = 0.0005, dropout=0.1)

    target_network = dueling_neural_network(path = '.testtest.zip', input_dims=14, fc1_dims=256,  fc2_dims=256, n_actions= 2,
                         loss = nn.MSELoss(), activation_function= [F.relu, F.relu], lr = 0.0005, dropout=0.1)
    return current_network, target_network

def set_up_dqn_networks():
    #path, input_dims, fc1_dims ,fc2_dims, n_actions, loss, activation_function, lr, dropout
    current_network = dqn_neural_network(path = '.test.zip', input_dims=14, fc1_dims=256,  fc2_dims=256, n_actions= 2,
                         loss = nn.MSELoss(), activation_function= [F.relu, F.relu], lr = 0.0005, dropout=0.1)

    target_network = dqn_neural_network(path = '.testtest.zip', input_dims=14, fc1_dims=256,  fc2_dims=256, n_actions= 2,
                         loss = nn.MSELoss(), activation_function= [F.relu, F.relu], lr = 0.0005, dropout=0.1)
    return current_network, target_network

def print_training_results(dqn, ddqn, dueling_dqn, dueling_ddqn):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    x = np.linspace(0,len(dqn), len(dqn))  
    y = dqn
    ax1.plot(x,y)
    x_1 = np.linspace(0,len(ddqn), len(ddqn))  
    y_1 = ddqn
    ax2.plot(x_1,y_1)  
    
    x_2 = np.linspace(0,len(dueling_dqn), len(dueling_dqn))  
    y_2 = dueling_dqn
    ax3.plot(x_2,y_2)

    x_3 = np.linspace(0,len(dueling_ddqn), len(dueling_ddqn))  
    y_3 = dueling_ddqn
    ax4.plot(x_3,y_3)  

    plt.show()

if __name__ == '__main__':
    current_network, target_network = set_up_dqn_networks()
    #strategy = double_q_learning
    epsilon = epsilon_greedy(eps_start= 0.99, eps_end = 0.03, eps_dec = 0.0001)
    action_space = [0, 1]
    replay = Replay_Buffer(max_size =100000, batch_size = 512, input_dims = 14)
    action = action_selection_q_learning(current_network, target_network, action_space)
    q_learning_agent = train_agent(action_space = action_space,
                        replace = 100,
                        epsilon = epsilon,
                        replay = replay 
                        )
    
    print("q learning")
    q_learning_agent.set_action_selection(action)
    q_learning_agent.set_networks(current_network, target_network)
    q_learning_agent.set_learning_strategy(learning, q_learning)
    train_results_dqn = q_learning_agent.train(training_steps = 2000, environment = env, print_results = True, average = 50)
    
    print("double q learning")
    current_network, target_network = set_up_dqn_networks()
    q_learning_agent.memory.reset_memory(input_dims = 14, batch_size = 512)
    q_learning_agent.reset_epsilon(0.99)
    q_learning_agent.set_networks(current_network, target_network)
    q_learning_agent.set_learning_strategy(learning, double_q_learning)
    train_results_ddqn = q_learning_agent.train(training_steps = 2000, environment = env, print_results = True, average = 50)

    print("dueling q networks")
    current_network, target_network = set_up_dueling_dqn_networks()
    q_learning_agent.set_networks(current_network, target_network)
    q_learning_agent.set_learning_strategy(learning, dueling_q_learning)
    q_learning_agent.memory.reset_memory(input_dims = 14, batch_size = 512)
    q_learning_agent.reset_epsilon(0.99)
    action = action_selection_dueling_q_learning(current_network, target_network, action_space)
    q_learning_agent.set_action_selection(action)
    train_results_dueling_dqn = q_learning_agent.train(training_steps = 2000, environment = env, print_results = True, average = 50)

    print("dueling double q networks")
    current_network, target_network = set_up_dueling_dqn_networks()
    q_learning_agent.set_networks(current_network, target_network)
    q_learning_agent.set_learning_strategy(learning, dueling_double_q_learning)
    q_learning_agent.memory.reset_memory(input_dims = 14, batch_size = 512)
    q_learning_agent.reset_epsilon(0.99)
    action = action_selection_dueling_q_learning(current_network, target_network, action_space)
    q_learning_agent.set_action_selection(action)
    train_results_dueling_double_dqn = q_learning_agent.train(training_steps = 2000, environment = env, print_results = True, average = 50)

    print_training_results(train_results_dqn, train_results_ddqn, train_results_dueling_dqn, train_results_dueling_double_dqn)