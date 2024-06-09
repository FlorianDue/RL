# some_file.py
#import sys
# caution: path[0] is reserved for script path (or '' in REPL)
#sys.path.insert(1, '../Environments')
import sys
sys.path.append('Agents')

from Neural_Networks.nn import neural_network
from Agents.agent import train_agent
from Environments.Order_Acceptance_and_Allocation import OAAP
from Agents.utils.learning_and_exploration import epsilon_greedy
from Agents.utils.replay_buffer import Replay_Buffer
from Agents.utils.learning_and_exploration import double_q_learning
import torch.nn as nn
import torch.nn.functional as F



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

#path, input_dims, fc1_dims ,fc2_dims, n_actions, loss, activation_function, lr, dropout
current_network = neural_network(path = '.test.zip', input_dims=14, fc1_dims=256,  fc2_dims=256, n_actions= 2,
                         loss = nn.MSELoss(), activation_function= [F.relu, F.relu], lr = 0.0001, dropout=0.1)

target_network = neural_network(path = '.testtest.zip', input_dims=14, fc1_dims=256,  fc2_dims=256, n_actions= 2,
                         loss = nn.MSELoss(), activation_function= [F.relu, F.relu], lr = 0.0001, dropout=0.1)

if __name__ == '__main__':

    #strategy = double_q_learning
    agent = train_agent(current_network = current_network,
                          target_network = target_network,
                          action_space = [0, 1],
                          replace = 50,
                          epsilon = epsilon_greedy(eps_start= 1, eps_end = 0.03, eps_dec = 1e-4),
                          replay = Replay_Buffer(max_size = 200000, batch_size = 512, input_dims = 14))
    
    agent.set_learning_strategy(double_q_learning)
    agent.train(training_steps = 5000, environment = env, print_results = True, average=50)