
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

        # print(self.scaler)

        # print('nn_dynamic_inputs', self.cfg.nn_dynamic_inputs)
        # print('nn_static_inputs', self.cfg.nn_static_inputs)
        # print('nn_outputs', self.concept_model.nn_outputs)

        self.input_size = len(self.concept_model.cfg.nn_dynamic_inputs) + len(self.concept_model.cfg.nn_static_inputs)
        self.output_size = len(self.concept_model.nn_outputs)
        self.hidden_layers = self.concept_model.cfg.hidden_layers

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
        basin_values = {}
        
        # Iterate over each basin
        for basin in xr_dataset['basin']:
            basin_data = {}
            
            # Iterate over each variable
            for var_name in xr_dataset.data_vars:
                var_values = xr_dataset[var_name].sel(basin=basin).values  # Get the variable's values for the current basin
                torch_values = torch.tensor(var_values, dtype=self.dtype)  # Convert to torch tensor
                basin_data[var_name] = torch_values  # Store the torch tensor in the basin's data
                
            basin_values[basin.item()] = basin_data  # Store the basin's data in the dictionary
        
        return basin_values


