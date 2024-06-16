import torch as T
import numpy as np

class epsilon_greedy():

    def __init__(self, eps_start, eps_end, eps_dec) -> None:
        self.epsilon = eps_start
        self.eps_finish = eps_end
        self.eps_dec = eps_dec
        self.descent = 1
        self.done =False

    def epsilon_dec_linear(self):
        if self.epsilon > self.eps_finish:
            self.epsilon -= self.eps_dec
        elif(self.done == False):
            print("epsilon descent completed")
            self.done = True

    def epsilon_dec_pow(self):
        if self.epsilon > self.eps_finish:
            self.epsilon = 1/pow(self.descent,2)
            self.descent += self.eps_dec
        elif(self.done == False):
            print("epsilon descent completed")
            self.done = True                    



class learning():
    def __init__(self, gamma, epsilon, replace_target, learning_step):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learn_step_ctr = 0
        self.replace_target = replace_target
        self.learning = learning_step()
        
    def replace_network(self, current_model, target_model):
        if(self.replace_target is not None and self.learn_step_ctr%self.replace_target == 0):
            return target_model.load_state_dict(current_model.state_dict())

    def learn(self, replay_buffer, current_model, target_model):
        #skip learning step in case that the replay buffer has not enough samples
        if replay_buffer.mem_counter < replay_buffer.batch_size:
            self.epsilon.epsilon_dec_linear()
            return
        self.replace_network(current_model, target_model)
        self.learning.learning_step(replay_buffer, current_model, target_model, self.gamma)
        self.learn_step_ctr+=1
        #perform epsilon descent
        self.epsilon.epsilon_dec_linear()
        return
    
class q_learning(): 

    def learning_step(self, replay_buffer, current_model, target_model, gamma):
        states, actions, rewards, new_states, dones = replay_buffer.sample_data(current_model)
        index = np.arange(replay_buffer.batch_size)
        q_curr = current_model.forward(states.long())[T.tensor(index).to(current_model.device).long(), actions.long()]
        q_tar = target_model.forward(new_states)
        best_actions = T.argmax(q_tar, dim=1)
        q_tar[dones] = 0.0
        #compute loss
        td_target = rewards + gamma*q_tar[index, best_actions]
        td_error = td_target - q_curr
        loss = T.mean(pow((td_error),2))
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.optimizer.step()

class double_q_learning(): 

    def learning_step(self, replay_buffer, current_model, target_model, gamma):        
        states, actions, rewards, new_states, dones = replay_buffer.sample_data(current_model)
        index = np.arange(replay_buffer.batch_size)
        q_curr = current_model.forward(states.long())[T.tensor(index).to(current_model.device).long(), actions.long()]
        q_next = current_model.forward(new_states)
        q_tar = target_model.forward(new_states)
        best_actions = T.argmax(q_next, dim=1)
        q_tar[dones] = 0.0
        td_target = rewards + gamma*q_tar[index, best_actions]
        td_error = td_target - q_curr
        loss = T.mean(pow(td_error,2))
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.optimizer.step()

class dueling_q_learning():        
    
    def learning_step(self, replay_buffer, current_model, target_model, gamma):        
        states, actions, rewards, new_states, dones = replay_buffer.sample_data(current_model)
        index = np.arange(replay_buffer.batch_size)
        
        A, V = current_model.forward(states)
        A_next, V_next = target_model.forward(new_states)
        q_curr = T.add(V, (A-A.mean(dim=1, keepdim=True)))[T.tensor(index).to(current_model.device).long(), actions.long()]
        q_tar = T.add(V_next, (A_next-A_next.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_tar[dones] = 0.0
        
        td_target = rewards + gamma*q_tar
        td_error = td_target-q_curr
        loss = T.mean(pow(td_error,2))
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.optimizer.step()

class dueling_double_q_learning():

    def learning_step(self, replay_buffer, current_model, target_model, gamma):
        states, actions, rewards, new_states, dones = replay_buffer.sample_data(current_model)
        index = np.arange(replay_buffer.batch_size)

        A, V = current_model.forward(states)
        q_curr = T.add(V, (A-A.mean(dim=1, keepdim = True)))[T.tensor(index).to(current_model.device).long(), actions.long()]

        A_curr_nex, V_curr_next = current_model.forward(new_states)
        q_curr_next = T.add(V_curr_next, (A_curr_nex-A_curr_nex.mean(dim=1, keepdim = True)))
        max_actions = T.argmax(q_curr_next, dim=1)

        A_tar_next, V_tar_next = target_model.forward(new_states)
        q_tar_next = T.add(V_tar_next, (A_tar_next-A_tar_next.mean(dim=1, keepdim = True)))[T.tensor(index).to(current_model.device).long(), max_actions.long()]
        q_tar_next[dones] = 0.0

        td_target = rewards + gamma*q_tar_next
        td_error = td_target - q_curr

        loss = T.mean(pow(td_error,2))
        current_model.optimizer.zero_grad()
        loss.backward()
        current_model.optimizer.step()


