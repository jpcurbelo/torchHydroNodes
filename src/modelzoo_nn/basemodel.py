
import torch
import torch.nn as nn
import xarray

from src.modelzoo_concept.basemodel import BaseConceptModel

class BaseNNModel(nn.Module):
    
    def __init__(self, concept_model: BaseConceptModel, ds_static: xarray.Dataset = None):
        super(BaseNNModel, self).__init__()

        self.concept_model = concept_model
        self.ds_static = ds_static
        
        # Extract quantities for easy access
        self.device = self.concept_model.cfg.device
        self.dtype = self.concept_model.cfg.precision['torch']
        self.scaler = self.concept_model.scaler

        # print('self.concept_model.cfg.nn_dynamic_inputs', self.concept_model.cfg.nn_dynamic_inputs)
        self.num_dynamic = len(self.concept_model.cfg.nn_dynamic_inputs) 
        self.num_static = len(self.concept_model.cfg.static_attributes)
        self.include_static = self.num_static > 0
        self.input_size = self.num_dynamic + self.num_static
        self.static_size = self.num_static if self.include_static else 0
        self.output_size = len(self.concept_model.nn_outputs)
        self.hidden_size = self.concept_model.cfg.hidden_size
        self.num_layers = len(self.hidden_size)
        self.dropout = self.concept_model.cfg.dropout

        # Compute mean and std for variables by basin
        self.torch_input_stds = self.xarray_to_torch(self.scaler['ds_feature_std'], variables=self.concept_model.cfg.nn_dynamic_inputs)
        self.torch_input_means = self.xarray_to_torch(self.scaler['ds_feature_mean'], variables=self.concept_model.cfg.nn_dynamic_inputs)
        # self.torch_input_mins = self.xarray_to_torch(self.scaler['ds_feature_min'], variables=self.concept_model.cfg.nn_mech_targets)

        # Target mean value
        self.torch_target_means = self.xarray_to_torch(self.scaler['ds_feature_mean'], variables=self.concept_model.cfg.target_variables)

        # print("self.torch_input_means", self.torch_input_means)
        # aux = input("Press Enter to continue...")

        if self.ds_static is not None:
            self.torch_static = self.xarray_to_torch(self.ds_static, variables=self.concept_model.cfg.static_attributes)
        else:
            self.torch_static = None

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

    def xarray_to_torch(self, xr_dataset, variables=None):
        '''
        Function to convert an xarray dataset to a dictionary of torch tensors
        
        - Args:
            - xr_dataset: xarray dataset object
            
        - Returns:
            - basin_values: Dictionary with the basin values as torch tensors
        '''
        if variables is None:
            variables = xr_dataset.data_vars.keys()

        basin_values = {}
        # Iterate over each basin
        for basin in xr_dataset['basin']:
            basin_data = []
            
            # Iterate over each variable
            for var_name in variables:

                var_value = xr_dataset[var_name.lower()].sel(basin=basin).values  # Get the variable's values for the current basin
                torch_value = torch.tensor(var_value, dtype=self.dtype)  # Convert to torch tensor
                basin_data.append(torch_value)  # Store the torch tensor in the basin's data
                
            basin_values[basin.item()] = basin_data  # Store the basin's data in the dictionary

            # List to torch tensor
            # print('self.device', self.device)
            basin_values[basin.item()] = torch.stack(basin_values[basin.item()], dim=0).reshape(1, -1).to(self.device)  
        
        return basin_values


