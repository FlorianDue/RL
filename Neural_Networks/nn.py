#import nn_cofig

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class dqn_neural_network(nn.Module):
    #lr = learning rate
    #dims = dict with {nbr_parameter, network_layer} 
    #path = path to store parameter
    #dropout, if true dropout layer

    def __init__(self, path, input_dims, fc1_dims ,fc2_dims, n_actions, loss, activation_function, lr, dropout):
        super(dqn_neural_network, self).__init__()
        self.path = path
        self.dropout = dropout
        self.loss = loss
        self.activation_function = activation_function

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn. Linear(fc1_dims, fc2_dims)
        self.dropout2 = nn.Dropout(p=dropout)
        self.Q = nn.Linear(fc2_dims, n_actions)
        #self.V=nn.Linear(fc2_dims, 1)


        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        #get training data to GPU
        self.device= self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #Ausgabe ob auf CPU oder GPU trainiert wird
        #if T.cuda.is_available:
        #    print('GPU')
        #else:
        #    print('CPU')
        self.to(self.device)

    def forward(self, state):
        state = state.type(T.float32)
        state = self.activation_function[0](self.fc1(state))
        state = self.activation_function[0](self.fc2(state))
        return self.Q(state)#, self.V(state)

    def save_params(self):
        T.save(self.state_dict(), self.path)

    def load_params(self):
        self.load_state_dict(T.load(self.path))  

class dueling_neural_network(nn.Module):
    #lr = learning rate
    #dims = dict with {nbr_parameter, network_layer} 
    #path = path to store parameter
    #dropout, if true dropout layer

    def __init__(self, path, input_dims, fc1_dims ,fc2_dims, n_actions, loss, activation_function, lr, dropout):
        super(dueling_neural_network, self).__init__()
        self.path = path
        self.dropout = dropout
        self.loss = loss
        self.activation_function = activation_function

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn. Linear(fc1_dims, fc2_dims)
        self.dropout2 = nn.Dropout(p=dropout)
        self.Q = nn.Linear(fc2_dims, n_actions)
        self.V=nn.Linear(fc2_dims, 1)


        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        #get training data to GPU
        self.device= self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #Ausgabe ob auf CPU oder GPU trainiert wird
        #if T.cuda.is_available:
        #    print('GPU')
        #else:
        #    print('CPU')
        self.to(self.device)

    def forward(self, state):
        state = state.type(T.float32)
        state = self.activation_function[0](self.fc1(state))
        state = self.activation_function[0](self.fc2(state))
        return self.Q(state), self.V(state)

    def save_params(self):
        T.save(self.state_dict(), self.path)

    def load_params(self):
        self.load_state_dict(T.load(self.path))          

