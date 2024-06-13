import xarray
import numpy as np
import torch
import torch.nn as nn

from src.utils.load_process_data import Config
from src.modelzoo_nn.basepretrainer import NNpretrainer

# Classes
class BaseHybridModel(nn.Module):

    def __init__(self,
                cfg: Config,
                pretrainer: NNpretrainer,
                ds: xarray.Dataset,
                scaler: dict,
                # odesmethod:str ='RK23'
                ):
        nn.Module.__init__(self)  # Initialize nn.Module
               
        self.cfg = cfg
        self.pretrainer = pretrainer
        self.dataset = ds
        self.scaler = scaler

        # Method to solve ODEs
        if hasattr(cfg, 'odesmethod'):
            self.odesmethod = cfg.odesmethod
        else:
            self.odesmethod = 'RK23'

        # Create the dataloader
        self.dataloader = self.pretrainer.create_dataloaders(is_trainer=True)
        self.num_batches = len(self.dataloader)

        # Device
        self.device = self.pretrainer.nnmodel.device

        # Set the data type attribute for the model
        self.data_type_np = cfg.precision['numpy']
        self.data_type_torch = cfg.precision['torch']

        # Time series
        self.time_series = np.linspace(0, len(self.dataset['date'].values) - 1, len(self.dataset['date'].values))

        # Epochs
        if hasattr(cfg, 'epochs'):
            self.epochs = cfg.epochs
        else:
            self.epochs = 100

        self.optimizer = self.pretrainer.optimizer
        self.scheduler = self.pretrainer.scheduler
    
    def forward(self, inputs, basin):
        '''This function should execute the model'''
        raise NotImplementedError("This function has to be implemented by the child class")

    def hybrid_model(self, t, y, *args):
        '''This function should execute the hybrid model (conceptual model + neural network model)'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def train(self):
        '''This function should train the model'''
        raise NotImplementedError("This function has to be implemented by the child class")