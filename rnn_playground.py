#!/usr/bin/env python3

# Heavily based on / slightly copied from:
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

import sys
import numpy as np
import torch as t
from torch import nn

assert sys.argv[1]
assert sys.argv[2]

dtype = np.float32
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

all_data = np.load(sys.argv[1])

data_count = len(all_data)
data_len = len(all_data[0])
data_width = len(all_data[0][0])

# Split off some data for testing
test_data_count = int(data_count / 5)
train_data = all_data[:-test_data_count]
test_data = all_data[-test_data_count:]

# Create input and target sequences
input_seq_train = t.Tensor([xs[:-1] for xs in train_data])
target_seq_train = t.Tensor([xs[1:] for xs in train_data])

input_seq_test = t.Tensor([xs[:-1] for xs in test_data])
target_seq_test = t.Tensor([xs[1:] for xs in test_data])

for x in [input_seq_train, target_seq_train, input_seq_test, target_seq_test]:
    x.to(device)


# Define our model
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


model = MyModel(data_width, data_width, 200, 4)
model.to(device)

# Define hyperparameters
n_epochs = 20
lr = 0.005

loss_criterion = nn.SmoothL1Loss()
optimizer = t.optim.Adam(model.parameters(), lr=lr)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    output, _ = model(input_seq_train)  # Run all the data through

    loss = loss_criterion(output.view(-1), target_seq_train.view(-1))
    loss.backward()
    optimizer.step()

    output, _ = model(input_seq_test)
    test_loss = loss_criterion(output.view(-1), target_seq_test.view(-1))

    if epoch % 5 == 0:
        print(
            "Epoch {}; training loss: {}  Testing loss: {}".format(
                epoch, loss.item(), test_loss.item()
            )
        )

t.save(model, sys.argv[2])
