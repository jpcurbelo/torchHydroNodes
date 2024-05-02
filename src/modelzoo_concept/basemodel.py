import numpy as np
import torch
import xarray

from src.utils.utils_load_process import Config
from src.datasetzoo.basedataset import BaseDataset

# Classes
class BaseConceptModel:
    
    def __init__(self, 
                 cfg: Config,
                 ds: xarray.Dataset,
                 odesmethod:str ='RK23'
                ):
        
        self.cfg = cfg
        self.ds = ds
        self.odesmethod = odesmethod
        
        # Set the data type attribute for the model
        self.data_type_np = cfg.precision['numpy']
        self.data_type_torch = cfg.precision['torch']
        
        # Interpolators
        self.interpolators = self.create_interpolator_dict()
        
    def conceptual_model(self, t, y, params):
        '''This function should implement the conceptual model to be used for the task'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def step_function(self, x):
        '''This function should implement the step function to be used for the task'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def create_interpolator_dict(self, t):
        '''This function should return the interpolator dictionary for the input variables'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def get_parameters(self):
        '''This function should return the model parameters'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def run (self, basin):
        '''This function should run the model'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def save_results(self, ds, results, basin, period='train'):
        '''This function should save the model results'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def plot_results(self, ds, q_bucket, basin, period='train', plot_prcp=False):
        '''This function should plot the model results'''
        raise NotImplementedError("This function has to be implemented by the child class")        
    
    # @property
    # def get_parameters(self):
    #     '''This function should return the model parameters'''
    #     raise NotImplementedError("This function has to be implemented by the child class")
    
    # @property
    # def get_input_variables(self):
    #     '''This function should return the input variables for the model'''
    #     raise NotImplementedError("This function has to be implemented by the child class")
    
    # @property
    # def get_output_variables(self):
    #     '''This function should return the output variables for the model'''
    #     raise NotImplementedError("This function has to be implemented by the child class")