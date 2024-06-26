import numpy as np
import torch
import xarray

from src.utils.load_process_data import Config
from src.datasetzoo.basedataset import BaseDataset

# Classes
class BaseConceptModel:
    
    def __init__(self, 
                 cfg: Config,
                 ds: xarray.Dataset,
                 scaler: dict,
                 odesmethod:str ='RK23'
                ):
        
        self.cfg = cfg
        self.dataset = ds
        self.scaler = scaler
        self.odesmethod = odesmethod
        
        # Set the data type attribute for the model
        self.data_type_np = cfg.precision['numpy']
        self.data_type_torch = cfg.precision['torch']
        
        # Time series
        self.time_series = np.linspace(0, len(ds['date'].values) - 1, len(ds['date'].values))
        
    def conceptual_model(self, t, y, params):
        '''This function should implement the conceptual model to be used for the task'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def step_function(self, x):
        '''This function shoulparams_dird implement the step function to be used for the task'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def run (self, basin):
        '''This function should run the model'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def save_results(self, ds, results, basin, period='train'):
        '''This function should save the model results'''
        raise NotImplementedError("This function has to be implemented by the child class")
  
    def shift_initial_states(self, start_and_end_dates, basin, period='valid'):
        '''This function should load the last states of the model in the previous period.
        To be used for the valid or test period after the train period'''
        raise NotImplementedError("This function has to be implemented by the child class")  