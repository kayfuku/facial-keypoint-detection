import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Model architecture
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers_sizes, drop_p=0.5):
        '''
            Builds a Multi-Layer Perceptron with arbitrary hidden layers.

            input_size: integer, size of the input layer
            outpu_size: integer, size of the output layer
            hidden_layers_sizes: list of integers, the sizes of the hidden layers

        '''
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers_sizes = hidden_layers_sizes

        # The first BatchNorm layer and the first layer of hidden layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(input_size)])
        self.lin_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers_sizes[0])])

        # All the rest of hidden layers
        layer_sizes = zip(hidden_layers_sizes[:-1], hidden_layers_sizes[1:])
        self.lin_layers.extend(nn.Linear(in_size, out_size) for in_size, out_size in layer_sizes)
        self.bn_layers.extend(nn.BatchNorm1d(size) for size in hidden_layers_sizes)

        self.dropout = nn.Dropout(p=drop_p)
        self.output = nn.Linear(hidden_layers_sizes[-1], output_size)

    def forward(self, x):
        ''' Forward pass through the network, returns the log softmax '''
        bn0 = self.bn_layers[0]
        x = bn0(x)

        for lin, bn in zip(self.lin_layers, self.bn_layers[1:]):
            x = F.relu(lin(x))
            x = self.dropout(x)
            x = bn(x)

        x = self.output(x)
        x = F.log_softmax(x, dim=1)

        return x
