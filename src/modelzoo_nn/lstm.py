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

    def forward(self, inputs, basin_list, use_grad=True):
        means = torch.stack([self.torch_input_means[b] for b in basin_list]).squeeze(1).to(inputs.device)
        stds = torch.stack([self.torch_input_stds[b] for b in basin_list]).squeeze(1).to(inputs.device)
        
        means = means.unsqueeze(1)
        stds = stds.unsqueeze(1)
        
        inputs = (inputs - means) / (stds + 1e-10)
        
        if use_grad:
            # print(f"Memory before lstm pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            lstm_out, _ = self.lstm(inputs)
            # print(f"Memory after lstm pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            lstm_out = self.dropout_layer(lstm_out)
            output = self.fc(lstm_out[:, -1, :])
            return output
        else:
            with torch.no_grad():  # Disable gradient calculation for inference
                # print(f"Memory before lstm pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                lstm_out, _ = self.lstm(inputs)
                # print(f"Memory after lstm pass: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                lstm_out = self.dropout_layer(lstm_out)
                output = self.fc(lstm_out[:, -1, :])
            return output