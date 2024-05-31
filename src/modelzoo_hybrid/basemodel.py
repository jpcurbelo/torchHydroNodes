import xarray
import numpy as np

from src.utils.load_process_data import Config
from src.modelzoo_nn.basepretrainer import NNpretrainer

# Classes
class BaseHybridModel:

    def __init__(self,
                 cfg: Config,
                 pretrainer: NNpretrainer,
                 ds: xarray.Dataset,
                #  scaler: dict,
                #  odesmethod:str ='RK23'
                ):
        
        self.cfg = cfg
        self.device = pretrainer.nnmodel.device
        self.pretrainer = pretrainer
        self.ds = ds

        # Basins
        self.basins = self.ds.basin.values

        # Set the data type attribute for the model
        self.data_type_np = cfg.precision['numpy']
        self.data_type_torch = cfg.precision['torch']

        # Time series
        self.time_series = np.linspace(0, len(ds['date'].values) - 1, len(ds['date'].values))

        # Interpolators
        self.interpolators = self.create_interpolator_dict()
    
    def train (self):
        # '''This function should train the model'''
        # raise NotImplementedError("This function has to be implemented by the child class")

        print('Training the hybrid model')