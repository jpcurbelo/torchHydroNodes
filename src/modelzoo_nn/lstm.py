import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel

class LSTMModel(BaseNNModel):

    def __init__(self, concept_model: BaseConceptModel, 
                 input_size: int, 
                 hidden_size: int, 
                 output_size: int, 
                 num_layers: int = 1, 
                 dropout: float = 0.0):
        super().__init__(concept_model)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Initialize the LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        # Initialize the fully connected layer
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # Initialize the dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inputs, basin_list):
        # Gather means and stds for the batch
        means = torch.stack([self.torch_input_means[b] for b in basin_list]).squeeze(1).to(inputs.device)
        stds = torch.stack([self.torch_input_stds[b] for b in basin_list]).squeeze(1).to(inputs.device)

        # Normalize the inputs
        inputs = (inputs - means) / (stds + 1e-10)

        # Pass through the LSTM layers
        lstm_out, (h_n, c_n) = self.lstm(inputs)

        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)

        # Pass through the fully connected layer
        output = self.fc(lstm_out[:, -1, :])  # Assuming we only want the output from the last time step

        return output