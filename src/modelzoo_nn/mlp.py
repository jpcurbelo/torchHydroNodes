
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel

class MLP(BaseNNModel):
    
    def __init__(self, concept_model: BaseConceptModel):
        super().__init__(concept_model)

    def create_layers(self):

        self.input_layer = nn.Linear(self.input_size, self.hidden_size[0])
        self.input_layer.name = 'input_layer'

        # Hidden Layers
        self.hidden = nn.ModuleList()
        for li, hidden in enumerate(self.hidden_size[:-1]):
            layer = nn.Linear(hidden, self.hidden_size[li+1])
            layer.name = f'hidden_layer_{li}'
            self.hidden.add_module(f'hidden{li+1}', layer)

        # Output Layer
        self.output_layer = nn.Linear(self.hidden_size[-1], self.output_size)
        self.output_layer.name = 'output_layer'

    def forward(self, inputs, basin_list):

        # Gather means and stds for the batch
        means = torch.stack([self.torch_input_means[b] for b in basin_list]).squeeze(1).to(inputs.device)
        stds = torch.stack([self.torch_input_stds[b] for b in basin_list]).squeeze(1).to(inputs.device)


        # print('inputs:', inputs)
        # print('means:', means)
        # print('stds:', stds)

        # Normalize the inputs
        inputs = (inputs - means) / stds

        # print('inputs:', inputs)

        # print('inputs:', inputs)
        # print('means:', means)
        # print('stds:', stds)

        # aux = input("Press Enter to continue...")

        # Pass through the input layer
        x = F.tanh(self.input_layer(inputs))

        # Hidden Layers
        for hidden in self.hidden:
            x = F.leaky_relu(hidden(x))
        
        # Output Layer
        x = self.output_layer(x)

        # # Clip negative values in the last dimension to 0
        # # x[:, -1] = F.relu(x[:, -1])
        # x = F.relu(x) + 1e-6

        # print('x:', x)

        return x

        


