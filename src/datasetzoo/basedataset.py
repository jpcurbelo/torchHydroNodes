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
        if 'nn_static_inputs' not in self.cfg or not self.cfg['nn_static_inputs']:
            self.cfg['nn_static_inputs'] = []
        
        # List of columns to keep, everything else will be removed to reduce memory footprint
        keep_cols = self.cfg['nn_dynamic_inputs'] + self.cfg['nn_static_inputs'] + self.cfg['target_variables']
        keep_cols = list(sorted(set(keep_cols)))
        # Lowercase all columns
        keep_cols = [col.lower() for col in keep_cols]
        
        for basin in tqdm(self.basins, file=sys.stdout):
            
            df = self._load_basin_data(basin)
            
            # Make the columns to be lower case
            df.columns = [col.lower() for col in df.columns]

            # Compute mean from min-max pairs if necessary
            df = self._compute_mean_from_min_max(df, keep_cols)
            
            # Remove unnecessary columns
            df = self._remove_unnecessary_columns(df, keep_cols)
            
            
            # # Make end_date the last second of the specified day, such that the
            # # dataset will include all hours of the last day, not just 00:00.
            # start_dates = self.start_and_end_dates[basin]["start_dates"]
            # end_dates = [
            #     date + pd.Timedelta(days=1, seconds=-1) for date in self.start_and_end_dates[basin]["end_dates"]
            # ]
            
            # basin_data_list = []
            # # Create xarray data set for each period slice of the specific basin
            # for i, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
            #     pass
            
            
            
            
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """This function has to return the data for the specified basin as a time-indexed pandas DataFrame"""
        raise NotImplementedError("This function has to be implemented by the child class")
    
    
    def _compute_mean_from_min_max(self, df, keep_cols):
        """
        Compute the mean from the minimum and maximum values for specified columns.

        Args:
            df (DataFrame): The DataFrame containing the data.
            keep_cols (list): A list of columns to keep.

        Returns:
            DataFrame: The DataFrame with mean columns added.

        Raises:
            KeyError: If no suitable min-max pairs are found for computing the mean.
        """
        if any(col.startswith('tmean') for col in keep_cols):
            # Loop through all available features and try to find min-max pairs
            for col in df.columns:
                if col.startswith(('tmax', 'tmin')):
                    complement_col = col.replace('tmax', 'tmin') if col.startswith('tmax') else col.replace('tmin', 'tmax')
                    mean_col_name = col.replace('max', 'mean').replace('min', 'mean')
                    
                    if complement_col in df.columns:
                        df[mean_col_name] = (df[col] + df[complement_col]) / 2
                        return df
                    else:
                        # If no min-max pairs found, raise KeyError
                        raise KeyError(f"Cannot compute '{mean_col_name}'. No suitable min-max pairs found.")

    def _remove_unnecessary_columns(self, df, keep_cols):
        """
        Remove unnecessary columns from DataFrame.

        Args:
            df (DataFrame): The DataFrame containing the data.
            keep_cols (list): A list of columns to keep.

        Returns:
            DataFrame: The DataFrame with unnecessary columns removed.

        Raises:
            KeyError: If any of the specified columns in keep_cols are not found in the DataFrame.

        """
        # Check if any of the specified columns are not found in the DataFrame
        not_available_columns = [col for col in keep_cols if not any(df_col.startswith(col) for df_col in df.columns)]
        if not_available_columns:
            msg = [
                f"The following features are not available in the data: {not_available_columns}. ",
                f"These are the available features: {df.columns.tolist()}"
            ]
            raise KeyError("".join(msg))
        
        # Keep only columns that start with the specified columns in keep_cols
        df = df[[col for col in df.columns if any(col.startswith(k) for k in keep_cols)]]
        
        return df

        
        
        
        
        
        