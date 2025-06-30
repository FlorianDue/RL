import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pygad

# genetic network
class genetic_network(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, n_actions, path):
        super(genetic_network, self).__init__()
        self.path = path
        # first hidden layer
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        # first dropout layer
        #self.dropout1 = nn.Dropout(p=0.1)
        # second hidden layer
        self.fc2 = nn. Linear(fc1_dims, fc2_dims)
        # second dropout layer
        #self.dropout2 = nn.Dropout(p=0.1)
        self.Q = nn.Linear(fc2_dims, n_actions)

        self.device = T.device("cpu")
        #"cuda" if T.cuda.is_available() else "cpu"
        self.to(T.device("cpu"))

    # forward computation of each network with the corresponding activation function

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x=F.relu(self.fc3(x))
        Q = self.Q(x)

        return Q

    # save model parameters
    def save_chkp(self):
        T.save(self.state_dict(), self.path)
    # load model parameters

    def load_chkp(self):
        self.load_state_dict(T.load(self.path))

class genetic_agent():
    
    def __init__(self, input_dims, n_actions, path, model):
        self.action_space = [i for i in range(n_actions)]
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.path = path
        self.model = model

     # action selection of the agent
    def choose_action(self, observation):
        # during the evaluation the agent always chooses its best known action
        state = T.tensor([observation], dtype=T.float).to(self.model.device)
        actions = self.model.forward(state)
        action = T.argmax(actions).item()
        return action

    # save parameters of q and target network
    def save_models(self, path):
        self.model.save_chkp()

    # load parameters of the q and target network
    def load_models(self, path):
        self.model.load_chkp()

    # save network structure which performed best in the validation data
    def current_model_structure(self, path):
        T.save(self.current_model.state_dict(), path)


class geneticalgorithm_interaction:

    def __init__(self, number_resources, env):
        self.number_resources = number_resources
        self.env = env

    #for the agent use a genetic_network inside a  
    def agent_interaction(self, env, orders, order_rewards, model, input_dims, n_actions, path):
        reward_list = np.zeros((self.number_resources, len(orders)), dtype=int)
        # Lists for plots and scores
        scores = []
        i = 0
        #create the agent to make decisions
        agent = genetic_agent(input_dims=input_dims,
                        n_actions=n_actions,  path=path, model=model)
        for i in range(len(orders)):
            done = False
            # reset environment
            observation = env.reset_env(static=True, static_orders=orders[i],
                                    static_rewards=order_rewards[i])
            score = 0
            reward_resources = np.zeros((n_actions-1, 1), dtype=int)
            while not done:
                # agent choose action
                action = agent.choose_action(observation)
                # perform action
                observation_, reward, done, info = env.step(action)
                # save the score
                score += reward
                # move to next state
                observation = observation_
                # save the reward the time step on corresponding resource
                if action > 0:
                    reward_resources[action-1] = reward_resources[action-1]+reward
                # save the reward of the episode
            for k in range(self.number_resources):
                # save reward of each resource
                if k > 0:
                    reward_list[k-1][i] = reward_resources[k-1]
                # save overall reward of the episode
                else:
                    reward_list[k-1][i] = score
            # save the score over the complete data set
            scores.append(score)
            i += 1
        # compute average score of each episode
        solution_fitness = np.sum(scores)/len(orders)
        return solution_fitness

class evaluation:

    def __init__(self, static_order, order_rewards, model, input_dims, n_actions, path, env, num_solutions, agent):
        self.static_order = static_order
        self.order_rewards = order_rewards 
        self.model = model
        self.input_dims = input_dims 
        self.n_actions = n_actions
        self.path = path
        self.env = env
        self.num_solutions = num_solutions
        self._last_data_gen = -1
        self.ga_instance = None
        self.agent = agent

    def fitness_func(self, instance, solution, sol_idx):
        #generate new evaluation data
        if self.ga_instance.generations_completed != self._last_data_gen:
            self.static_order, self.order_rewards = self.env.create_static_data()
            self._last_data_gen = self.ga_instance.generations_completed
        #convert model weights from vector to dictionary
        model_weights_matrix = pygad.torchga.model_weights_as_dict(model=self.model,weights_vector=solution)
        #load the parameters of each individual
        self.model.load_state_dict(model_weights_matrix)
        #compute fitness of each individual
        #static = False, static_orders= None, static_rewards = False, idx = None
        solution_fitness = self.agent.agent_interaction(self.env, self.static_order, self.order_rewards, self.model, self.input_dims, self.n_actions, self.path)
        #exchange evaluation data
        return solution_fitness

    def callback_generation(self, ga_instance):
        self.ga_instance = ga_instance 
        print("Generation = {generation}".format(
            generation = self.ga_instance.generations_completed))
        fitnesses = self.ga_instance.last_generation_fitness
        print(f"Best Fitness (this gen) = {max(fitnesses):.5f}")

    def register_instance(self, ga_instance):
        self.ga_instance = ga_instance
