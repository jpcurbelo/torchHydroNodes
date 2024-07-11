import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel

class LSTM(BaseNNModel):

    def __init__(self, concept_model: BaseConceptModel):
        super().__init__(concept_model)

    def create_layers(self):

        # Initialize the LSTM layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size[0], num_layers=self.num_layers, 
                            dropout=self.dropout, batch_first=True)

        # Initialize the fully connected layer
        self.fc = nn.Linear(self.hidden_size[-1], self.output_size)

        # Initialize the dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, inputs, basin_list):
        
        # print(f"Memory usage before normalization: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        means = torch.stack([self.torch_input_means[b] for b in basin_list]).squeeze(1).to(inputs.device)
        stds = torch.stack([self.torch_input_stds[b] for b in basin_list]).squeeze(1).to(inputs.device)

        means = means.unsqueeze(1)
        stds = stds.unsqueeze(1)

        inputs = (inputs - means) / (stds + 1e-10)
        # print(f"Memory usage after normalization: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        lstm_out, _ = self.lstm(inputs)
        # print(f"Memory usage after LSTM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        lstm_out = self.dropout_layer(lstm_out)
        # print(f"Memory usage after dropout: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        output = self.fc(lstm_out[:, -1, :])
        # print(f"Memory usage after fully connected layer: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # del means, stds, lstm_out #, h_n, c_n
        # torch.cuda.empty_cache()
        # print(f"Memory usage after clearing cache: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # aux = input("Press Enter to continue...")

        return output