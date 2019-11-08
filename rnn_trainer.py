#!/usr/bin/env python3

# Heavily based on / slightly copied from:
# https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/

import sys
import numpy as np
import torch as t
from torch import nn

from my_model import MyModel

assert sys.argv[1]
assert sys.argv[2]

dtype = np.float32
device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

all_data = np.load(sys.argv[1])
np.random.shuffle(all_data)

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


model = MyModel(data_width, data_width, 200, 4)
model.to(device)

# Define hyperparameters
n_epochs = 30
lr = 0.01

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

    if epoch % 1 == 0:
        print(
            "Epoch {}; training loss: {}  Testing loss: {}".format(
                epoch, loss.item(), test_loss.item()
            )
        )

params = [data_width, data_width, 200, 4]

t.save({"state_dict": model.state_dict(), "params": params}, sys.argv[2])
