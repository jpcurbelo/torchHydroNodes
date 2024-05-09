
import torch
import torch.nn as nn

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel

class MLP(BaseNNModel):
    
    def __init__(self, concept_model: BaseConceptModel):
        super().__init__(concept_model)


        # self.input_size = len(cfg.nn_dynamic_inputs) + len(cfg.nn_static_inputs)
        # self.output_size = len(cfg.nn_outputs)

        # self.hidden_layers = hidden_layers
        # self.input_means = torch.tensor(input_means, dtype=DTYPE).to(DEVICE)
        # self.input_stds = torch.tensor(input_stds, dtype=DTYPE).to(DEVICE)
        # self.submodelID = submodelID

    def create_layers(self):

        self.input_layer = nn.Linear(self.input_size, self.hidden_layers[0])
        self.input_layer.name = 'input_layer'

        # Hidden Layers
        self.hidden_layers = nn.ModuleList()
        for li, hidden in enumerate(self.hidden_layers[:-1]):
            layer = nn.Linear(hidden, self.hidden_layers[li+1])
            layer.name = f'hidden_layer_{li}'
            self.hidden.add_module(f'hidden{li+1}', layer)

        # Output Layer
        self.output_layer = nn.Linear(self.hidden_layers[-1], self.output_size)
        self.output_layer.name = 'output_layer'

    # def forward(self, inputs, basin):
        
    #     # Normalize the inputs
    #     inputs = (inputs - self.torch_input_means[basin]) / self.torch_input_stds[basin]
        


