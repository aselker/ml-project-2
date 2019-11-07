import torch as t
from torch import nn


class MyModel(nn.Module):
    def __init__(self, input_size, output_size, state_size, n_layers):
        super(MyModel, self).__init__()
        self.state_size = state_size
        self.n_layers = n_layers

        # Recurrent layer
        self.rnn = nn.RNN(input_size, state_size, n_layers, batch_first=True)
        # FC layer
        self.fc = nn.Linear(state_size, output_size)

    def forward(self, x, state=None):
        batch_size = x.size(0)

        if state is None:
            # Initialize the state to zeros
            state = t.zeros(self.n_layers, batch_size, self.state_size)

        # Run the RNN, producing outputs and a final state (I think)
        out, state = self.rnn(x, state)

        # Reshape and process the outputs
        out = out.contiguous().view(-1, self.state_size)
        out = self.fc(out)

        return out, state
