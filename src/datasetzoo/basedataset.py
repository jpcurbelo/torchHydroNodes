from torch.utils.data import Dataset
from typing import Dict, Union
import xarray
import pandas as pd
import logging
from tqdm import tqdm
import sys

from src.utils.utils_load_process import load_basin_file

LOGGER = logging.getLogger(__name__)


# Ref -> https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/datasetzoo/basedataset.py
class BaseDataset(Dataset):
    
    def __init__(self,
            cfg: dict,
            is_train: bool = True,
            basin: str = None,
            scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        
            
        if not is_train and not scaler:
            raise ValueError("During evaluation of validation or test period, scaler dictionary has to be passed")

            # if cfg.use_basin_id_encoding and not id_to_int:
            #     raise ValueError("For basin id embedding, the id_to_int dictionary has to be passed anything but train")
        
        self.scaler = scaler
        
        self.basins = load_basin_file(cfg['basin_file'])
        
        # # During training we log data processing with progress bars, but not during validation/testing
        # self._disable_pbar = cfg.verbose == 0 or not self.is_train
        
        # Initialize class attributes that are filled in the data loading functions
        self._x_d = {}
        self._x_h = {}
        self._x_f = {}
        self._x_s = {}
        self._attributes = {}
        self._y = {}
        self._per_basin_target_stds = {}
        self._dates = {}
        self.start_and_end_dates = {}
        self.num_samples = 0
        self.period_starts = {}  # needed for restoring date index during evaluation
        
        # Load and preprocess data
        self._load_data()
    
    def __len__(self):
        return self.num_samples
    
    
    def _load_data(self):
        
        # # Load attributes first to sanity-check those features before doing the compute expensive time series loading
        # self._load_combined_attributes()
        
        xr = self._load_or_create_xarray_dataset()
        
        print('xr', xr)
        
    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        
        data_list = []
        
        # Check if static inputs are provided - it is assumed that dynamic inputs and target are always provided
        print('nn_static_inputs' not in self.cfg, 'nn_static_inputs' not in self.cfg.keys())
        if 'nn_static_inputs' not in self.cfg or not self.cfg['nn_static_inputs']:
            self.cfg['nn_static_inputs'] = []
        
        # List of columns to keep, everything else will be removed to reduce memory footprint
        keep_cols = self.cfg['nn_dynamic_inputs'] + self.cfg['nn_static_inputs'] + self.cfg['target_variables']
        keep_cols = list(sorted(set(keep_cols)))
        
        for basin in tqdm(self.basins, file=sys.stdout):
            
            df = self._load_basin_data(basin)
            
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """This function has to return the data for the specified basin as a time-indexed pandas DataFrame"""
        raise NotImplementedError("This function has to be implemented by the child class")
        
        
        
        
        
        