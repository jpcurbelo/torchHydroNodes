import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray
import numpy as np

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel

class LSTM(BaseNNModel):

    def __init__(self, concept_model: BaseConceptModel, ds_static: xarray.Dataset = None):
        super().__init__(concept_model, ds_static)

    def create_layers(self):

        # Initialize the input gate linear layer for static inputs
        self.input_gate = nn.Linear(self.num_static, self.hidden_size[0])     

        # Initialize the LSTM layers
        self.lstm = nn.LSTM(input_size=self.num_dynamic, 
                            hidden_size=self.hidden_size[0], 
                            num_layers=self.num_layers, 
                            dropout=self.dropout if self.num_layers > 1 else 0, 
                            batch_first=True)  

        # Initialize the dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

        # Initialize the fully connected layer
        self.fc = nn.Linear(self.hidden_size[-1], self.output_size)

    def forward(self, dynamic_inputs, basin_list, static_inputs=None, use_grad=True):
        means = torch.stack([self.torch_input_means[b] for b in basin_list]).squeeze(1).to(dynamic_inputs.device)
        stds = torch.stack([self.torch_input_stds[b] for b in basin_list]).squeeze(1).to(dynamic_inputs.device)
        
        means = means.unsqueeze(1)
        stds = stds.unsqueeze(1)
        dynamic_inputs = (dynamic_inputs - means) / (stds + np.finfo(float).eps)

        if static_inputs is not None:
            # Compute the input gate activations using static inputs
            input_gate_activations = torch.sigmoid(self.input_gate(static_inputs))
        else:
            input_gate_activations = None
        
        if use_grad:
            lstm_out, _ = self.lstm(dynamic_inputs)
            if input_gate_activations is not None:
                lstm_out = lstm_out * input_gate_activations.unsqueeze(1)  # Apply input gate activations
            lstm_out = self.dropout_layer(lstm_out)
            output = self.fc(lstm_out[:, -1, :])
        else:
            with torch.no_grad():  # Disable gradient calculation for inference
                lstm_out, _ = self.lstm(dynamic_inputs)
                if input_gate_activations is not None:
                    lstm_out = lstm_out * input_gate_activations.unsqueeze(1)
                lstm_out = self.dropout_layer(lstm_out)
                output = self.fc(lstm_out[:, -1, :])

        # # Print the memory usage on the GPU
        # allocated = torch.cuda.memory_allocated()
        # reserved = torch.cuda.memory_reserved()
        # print(use_grad, f"Memory Allocated: {allocated / (1024 ** 2):.2f} MB")
        # print(use_grad, f"Memory Reserved: {reserved / (1024 ** 2):.2f} MB")

        
        # # Retrieve the minimum values for the basins
        # min_values = torch.stack([self.torch_input_mins[b] for b in basin_list]).squeeze(1).to(dynamic_inputs.device)
        # # Clip the outputs
        # output = torch.maximum(output, min_values)

        # # # Clear the cache
        # # torch.cuda.empty_cache()
        
        return output
    