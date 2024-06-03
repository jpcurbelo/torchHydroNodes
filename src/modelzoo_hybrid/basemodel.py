import xarray
import numpy as np

from src.utils.load_process_data import Config
from src.modelzoo_nn.basepretrainer import NNpretrainer

# Classes
class BaseHybridModel():

    def __init__(self,
                 cfg: Config,
                 pretrainer: NNpretrainer,
                 ds: xarray.Dataset,
                #  scaler: dict,
                # odesmethod:str ='RK23'
                ):
        
        self.cfg = cfg
        self.pretrainer = pretrainer
        self.ds = ds

        # Method to solve ODEs
        if hasattr(self.cfg, 'odesmethod'):
            self.odesmethod = self.cfg.odesmethod
        else:
            self.odesmethod = 'RK23'

        # Create the dataloader
        self.dataloader = self.pretrainer.create_dataloaders()
        self.num_batches = len(self.dataloader)

        # Device
        self.device = self.pretrainer.nnmodel.device

        # # Basins
        # self.basins = self.ds.basin.values

        # Set the data type attribute for the model
        self.data_type_np = cfg.precision['numpy']
        self.data_type_torch = cfg.precision['torch']

        # Time series
        self.time_series = np.linspace(0, len(ds['date'].values) - 1, len(ds['date'].values))

        # # Interpolators
        # self.interpolators = self.create_interpolator_dict()

        # # Input/output variables to NN model
        # self.nn_input_var_names = self.cfg.nn_dynamic_inputs
        # self.nn_output_var_names = self.cfg.nn_mech_targets

        # Epochs
        if hasattr(self.cfg, 'epochs'):
            self.epochs = self.cfg.epochs
        else:
            self.epochs = 100

        # # Number of workers
        # if hasattr(self.cfg, 'num_workers'):
        #     self.num_workers = self.cfg.num_workers
        # else:
        #     self.num_workers = 8

        self.optimizer = self.pretrainer.optimizer
    
    def forward(self, inputs, basin):
        '''This function should execute the model'''
        raise NotImplementedError("This function has to be implemented by the child class")

    def hybrid_model(self, t, y, *args):
        '''This function should execute the hybrid model (conceptual model + neural network model)'''
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def train(self):
        '''This function should train the model'''
        raise NotImplementedError("This function has to be implemented by the child class")