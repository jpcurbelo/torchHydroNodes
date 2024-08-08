import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel

class MLP(BaseNNModel):
    
    def __init__(self, concept_model: BaseConceptModel, ds_static: xarray.Dataset = None):
        super().__init__(concept_model, ds_static)

    def create_layers(self):

        self.input_layer = nn.Linear(self.input_size, self.hidden_size[0])
        self.input_layer.name = 'input_layer'

        # Hidden Layers
        self.hidden = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for li, hidden in enumerate(self.hidden_size[:-1]):
            layer = nn.Linear(hidden, self.hidden_size[li+1])
            layer.name = f'hidden_layer_{li}'
            self.hidden.add_module(f'hidden{li+1}', layer)
            # Initialize dropout layer
            self.dropouts.add_module(f'dropout{li+1}', nn.Dropout(self.dropout))

        # Output Layer
        self.output_layer = nn.Linear(self.hidden_size[-1], self.output_size)
        self.output_layer.name = 'output_layer'

    def forward(self, dynamic_inputs, basin_list, static_inputs=None, use_grad=True):

        # Gather means and stds for the batch
        means = torch.stack([self.torch_input_means[b] for b in basin_list]).squeeze(1).to(dynamic_inputs.device)
        stds = torch.stack([self.torch_input_stds[b] for b in basin_list]).squeeze(1).to(dynamic_inputs.device)

        # Normalize the dynamic inputs
        dynamic_inputs = (dynamic_inputs - means) / (stds + 1e-10)

        # Concatenate static inputs if included
        if self.include_static and static_inputs is not None:
            # Expand static inputs to match the batch size
            static_inputs = static_inputs.expand(dynamic_inputs.shape[0], -1)
            inputs = torch.cat((dynamic_inputs, static_inputs), dim=1)
        else:
            inputs = dynamic_inputs

        if use_grad:
            # Pass through the input layer
            x = F.tanh(self.input_layer(inputs))
            # Hidden Layers
            for hidden, dropout in zip(self.hidden, self.dropouts):
                x = F.leaky_relu(hidden(x))
                x = dropout(x)
            # Output Layer
            x = self.output_layer(x)
        else:
            with torch.no_grad():
                # Pass through the input layer
                x = F.tanh(self.input_layer(inputs))
                # Hidden Layers
                for hidden, dropout in zip(self.hidden, self.dropouts):
                    x = F.leaky_relu(hidden(x))
                    x = dropout(x)
                # Output Layer
                x = self.output_layer(x)

        # # Retrieve the minimum values for the basins
        # min_values = torch.stack([self.torch_input_mins[b] for b in basin_list]).squeeze(1).to(dynamic_inputs.device)
        
        # # Clip the outputs
        # x = torch.maximum(x, min_values)

        return x
