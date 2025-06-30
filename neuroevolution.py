from Environments.Order_Acceptance_and_Allocation import OAAP
from Agents.ga_agent import geneticalgorithm_interaction, genetic_network, evaluation
import numpy as np
import pygad
import pygad.torchga as torchga
import torch as T
import matplotlib.pyplot as plt

production_lines = 1
env = OAAP(episode_length = 20,
    production_lines = production_lines,
    nbr_resources_per_line = 6,
    resource_capacity = 6,
    min_rev = 1,
    max_rev = 9,
    action_space = [0, 1],
    order_set = None,
    penalty_reward = 20,
    len_static_data = 5)

agent = geneticalgorithm_interaction(production_lines, env)


# define the model
input_dims = [14]
fc1_dims = 64
fc2_dims = 64
# save model
path = "network_genetic_algorithm.zip"
# possible actions (rejection or acceptance)
n_actions = 2
model = genetic_network(input_dims, fc1_dims, fc2_dims, n_actions, path)



#define the GA approach
num_solutions = 50
torch_ga = torchga.TorchGA(model=model, num_solutions=num_solutions)
initial_population = torch_ga.population_weights
num_parents_mating = 25
num_generations = 10
parent_selection_type = "tournament"
crossover_type = "single_point"  # Type of the crossover operator
mutation_type = "random"
training_time = []
crossover_values = 0.25
mutation_values = 30
training_scores = np.zeros((2, int(num_generations)+2))
save_paths = ["network_genetic_algorithm_exp8.zip"]
ctr = 0

#create initial data set and define a evaluation classfor the training
static_order, order_rewards = env.create_static_data()
train = evaluation(static_order, order_rewards, model, input_dims, n_actions, path, env, num_solutions, agent) 



#train an agent
ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=num_solutions,
                           initial_population=initial_population,
                           fitness_func= train.fitness_func,
                           crossover_type=crossover_type,
                           parent_selection_type=parent_selection_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_values,
                           keep_parents=0,
                           on_generation=train.callback_generation,
                           save_best_solutions=False)
#register the ga instance in the evaluation class
train.register_instance(ga_instance)
ga_instance.run()


# Returning the details of the best solution.
results = ga_instance.best_solutions_fitness
    #return training data
solution, solution_fitness, solution_idx = ga_instance.best_solution()
results.append(solution_fitness)
print('results', results)
#save the network parameters
model = genetic_network(input_dims, fc1_dims, fc2_dims, n_actions, path)
model_weights_matrix = pygad.torchga.model_weights_as_dict(
        model=model, weights_vector=solution)
T.save(model_weights_matrix, path)


x = np.linspace(1,len(results),len(results))

plt.figure(figsize=(10,5))
plt.plot(x, results, color = 'black',linewidth =1, label = 'Agent')
plt.xlabel('Generations',fontsize=12)
plt.ylabel('Average Reward',fontsize=12)
plt.grid(True)
plt.show()