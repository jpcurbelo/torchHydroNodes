
import torch
import torch.nn as nn

from src.modelzoo_concept.basemodel import BaseConceptModel

class BaseNNModel(nn.Module):
    
    def __init__(self, concept_model: BaseConceptModel):
        super(BaseNNModel, self).__init__()

        self.concept_model = concept_model
        
        # Extract quantities for easy access
        self.device = self.concept_model.cfg.device
        self.dtype = self.concept_model.cfg.precision['torch']
        self.scaler = self.concept_model.scaler

        self.input_size = len(self.concept_model.cfg.nn_dynamic_inputs) + len(self.concept_model.cfg.nn_static_inputs)
        self.output_size = len(self.concept_model.nn_outputs)
        self.hidden_size = self.concept_model.cfg.hidden_size

        # Compute mean and std for variables by basin
        self.torch_input_stds = self.xarray_to_torch(self.scaler['ds_feature_std'])
        self.torch_input_means = self.xarray_to_torch(self.scaler['ds_feature_mean'])

        # Create the NN model
        self.create_layers()

        # Move model to the specified device
        self.to(self.device) 

    def create_layers(self):
        '''This function should create the layers of the neural network'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def forward(self, inputs, basin):
        '''This function should implement the forward pass of the neural network'''
        raise NotImplementedError("This function has to be implemented by the child class")

    def xarray_to_torch(self, xr_dataset):
        '''
        Function to convert an xarray dataset to a dictionary of torch tensors
        
        - Args:
            - xr_dataset: xarray dataset object
            
        - Returns:
            - basin_values: Dictionary with the basin values as torch tensors
        '''

        basin_values = {}
        # Iterate over each basin
        for basin in xr_dataset['basin']:
            basin_data = []
            
            # Iterate over each variable
            for var_name in self.concept_model.cfg.nn_dynamic_inputs:

                var_value = xr_dataset[var_name.lower()].sel(basin=basin).values  # Get the variable's values for the current basin
                torch_value = torch.tensor(var_value, dtype=self.dtype)  # Convert to torch tensor
                basin_data.append(torch_value)  # Store the torch tensor in the basin's data
                
            basin_values[basin.item()] = basin_data  # Store the basin's data in the dictionary

            # List to torch tensor
            basin_values[basin.item()] = torch.stack(basin_values[basin.item()], dim=0).reshape(1, -1).to(self.device)  
        
        return basin_values


