#!/usr/bin/env python3

# Heavily based on / slightly copied from:
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

import numpy as np
import torch as t
from torch import nn

dtype = np.float32
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

data_count = 30
data_len = 1000
mod = 10  # Maximum size of a data number thing

# Make some training data
train_data = [
    [np.random.choice(range(10)) for _ in range(2)] for _ in range(data_count)
]
while len(train_data[0]) < data_len:
    for d in train_data:
        d.append((d[-1] + d[-2]) % mod)


def one_hot(x):
    y = np.zeros(mod, dtype=dtype)
    y[x] = 1
    return y


input_seq = t.Tensor([[one_hot(x) for x in xs[:-1]] for xs in train_data])
target_classes = t.Tensor([xs[1:] for xs in train_data])
target_seq = t.Tensor([[one_hot(x) for x in xs[1:]] for xs in train_data])
input_seq.to(device)
target_seq.to(device)


class MyModel(nn.Module):
    def __init__(self, input_size, output_size, state_size, n_layers):
        super(MyModel, self).__init__()
        self.state_size = state_size
        self.n_layers = n_layers

        # Recurrent layer
        self.rnn = nn.RNN(input_size, state_size, n_layers, batch_first=True)
        # FC layer
        self.fc = nn.Linear(state_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)

        # Initialize the state to zeros
        state = t.zeros(self.n_layers, batch_size, self.state_size)

        # Run the RNN, producing outputs and a final state (I think)
        out, state = self.rnn(x, state)

        # Reshape and process the outputs
        out = out.contiguous().view(-1, self.state_size)
        out = self.fc(out)

        return out, state


model = MyModel(mod, mod, 20, 1)
model.to(device)

# Define hyperparameters
n_epochs = 100
lr = 0.01

loss_criterion = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    output, state = model(input_seq)  # Run all the data through
    # loss = loss_criterion(output, target_seq.long().view(-1, mod))
    loss = loss_criterion(output, target_classes.long().view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print("Epoch {}; loss {}".format(epoch, loss.item()))
