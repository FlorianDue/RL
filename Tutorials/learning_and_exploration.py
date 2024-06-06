import numpy as np
import torch as T
class epsilon_greedy():

    def __init__(self, eps_start, eps_finish, eps_dec) -> None:
        self.epsilon = eps_start
        self.eps_finish = eps_finish
        self.eps_dec = eps_dec

    def epsilon_dec_linear(self):
        if self.epsilon > self.eps_finish:
            self.epsilon -= self.eps_dec

    #def epsilon_dec_cubic(self):
    #    if self.epsilon > self.eps_finish:

class q_learning():

    def __init__(self, gamma, epsilon, replace_target):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learn_step_ctr = 0
        self.replace_target = replace_target 

    def replace_network(self, current_model, target_model):
        if(self.replace_target is not None and self.learn_step_ctr%self.replace_target == 0):
            return target_model.load_state_dict(current_model.state_dict())


    def learn(self, replay_buffer, current_model, target_model):
        #skip learning step in case that the replay buffer has not enough samples
        if replay_buffer.mem_counter < replay_buffer.batch_size:
            self.epsilon.epsilon_dec_linear()
            return
        self.replace_network(current_model, target_model)
        states, actions, rewards, new_states, dones = replay_buffer.sample_data(current_model)
        index = T.arange(replay_buffer.batch_size, dtype = T.float32)
        q_curr = current_model.forward(states.to(T.long))[index.to(T.long), actions.to(T.long)]
        q_tar = target_model.forward(new_states.to(T.long))
        best_actions = T.argmax(q_tar, dim=1)
        q_tar[dones] = 0.0
        #compute loss
        td_target = rewards + self.gamma*q_tar[index.to(T.long), best_actions.to(T.long)]
        loss = current_model.loss(td_target, q_curr)
        #optimize network weights
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.optimizer.step()
        #perform epsilon descent
        self.epsilon.epsilon_dec_linear()
        return

