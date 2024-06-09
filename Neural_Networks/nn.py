#import nn_cofig

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class neural_network(nn.Module):
    #lr = learning rate
    #dims = dict with {nbr_parameter, network_layer} 
    #path = path to store parameter
    #dropout, if true dropout layer

    def __init__(self, path, input_dims, fc1_dims ,fc2_dims, n_actions, loss, activation_function, lr, dropout):
        super(neural_network, self).__init__()
        self.path = path
        self.dropout = dropout
        self.loss = loss
        self.activation_function = activation_function

        #first hidden layer
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        #first dropout layer
        self.dropout1 = nn.Dropout(p=dropout)
        #second hidden layer
        self.fc2 = nn. Linear(fc1_dims, fc2_dims)
        #second dropout layer
        self.dropout2 = nn.Dropout(p=dropout)
        #output stream to compute the advantage function
        self.Q = nn.Linear(fc2_dims, n_actions)
        #output stream to compute the state-value function
        self.V=nn.Linear(fc2_dims, 1)


        self.optimizer = optim.RMSprop(self.parameters(), lr = lr)
        #get training data to GPU
        self.device=T.device('cpu')
        self.to(T.device('cpu'))

    def forward(self, state):
        state = state.type(T.float32)
        x=F.relu(self.fc1(state))
        x=F.relu(self.fc2(x))
        #x=F.relu(self.fc3(x))
        return self.Q(x)

    def save_params(self):
        T.save(self.state_dict(), self.path)

    def load_params(self):
        self.load_state_dict(T.load(self.path))  

