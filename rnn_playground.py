#!/usr/bin/env python3

# Heavily based on / slightly copied from:
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

import numpy as np
import torch as t
from torch import nn

dtype = np.float32

data_len = 200
mod = 10  # Maximum size of a data number thing

# Make some training data
train_data = [[1, 1], [1, 3], [2, 4]]
while len(train_data[0]) < data_len:
    for d in train_data:
        d.append((d[-1] + d[-2]) % mod)


def one_hot(x):
    y = np.zeros(mod, dtype=dtype)
    y[x] = 1
    return y


input_seq = t.Tensor([[one_hot(x) for x in xs[:-1]] for xs in train_data])
target_seq = t.Tensor([[one_hot(x) for x in xs[1:]] for xs in train_data])


# Do we have a CUDA device?
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MyModel(nn.Module):
    def __init__(self, input_size, output_size, state_size, n_layers):
        super(Model, self).__init__()
        self.state_size = state_size
        self.n_layers = n_layers

        # Recurrent layer
        self.rnn = nn.RNN(input_size, state_size, n_layers, batch_first=True)
        # FC layer
        self.fc = nn.Linear(state_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize the state to zeros
        state = torch.zeros(self.n_layers, batch_size, self.state_size)

        # Run the RNN, producing outputs and a final state (I think)
        out, state = self.rn(x, state)

        # Reshape and process the outputs
        out = out.contiguous().view(-1, self.state_size)
        out = self.fc(out)

        return out, state
