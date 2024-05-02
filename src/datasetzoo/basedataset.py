from torch.utils.data import Dataset
from typing import Dict, Union
import xarray
import pandas as pd
import logging
from tqdm import tqdm
import sys
import numpy as np
import torch

from src.utils.utils_load_process import (
    Config, 
    load_basin_file,
)

# Ref -> https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/datasetzoo/basedataset.py
class BaseDataset(Dataset):
    
    def __init__(self,
            cfg: Config,
            is_train: bool = True,
            scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = dict()):
        
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        self._compute_scaler = True if is_train else False
        
        if not is_train and not scaler:
            raise ValueError("During evaluation of validation or test period, scaler dictionary has to be passed")
        self.scaler = scaler
        
        self.basins = load_basin_file(cfg.basin_file)
        
        # During training we log data processing with progress bars, but not during validation/testing
        self._disable_pbar = not self.cfg.verbose
        
        # Initialize class attributes that are filled in the data loading functions
        self._x_d = dict()
        self._x_h = dict()
        self._x_f = dict()
        self._x_s = dict()
        self._attributes = dict()
        self._y = dict()
        self._per_basin_target_stds = dict()
        self._dates = dict()
        self.start_and_end_dates = dict()
        self.num_samples = 0
        self.period_starts = dict()  # needed for restoring date index during evaluation
        
        # Get the start and end dates for each period
        self._get_start_and_end_dates()
    
        # Load and preprocess data
        self._load_data()
    
    def __len__(self):
        return self.num_samples
    
    def _get_start_and_end_dates(self):
        '''
        Extract the minimum (start) and maximum (end) dates across all train, validation, and test periods.
        
        - Returns:
            min_date: datetime, minimum date.
        '''
        
        period_list = ['train', 'valid'] if self.is_train else ['test']
        for period in period_list:
            start_key = f'{period}_start_date'
            end_key = f'{period}_end_date'
            
            # Check if start and end dates are properties of the Config object
            if hasattr(self.cfg, '_cfg') and start_key in self.cfg._cfg and end_key in self.cfg._cfg:
                start_date = getattr(self.cfg, start_key) 
                end_date   = getattr(self.cfg, end_key)
                
                self.start_and_end_dates[period] = {
                    'start_date': start_date,
                    'end_date': end_date
                }

        # min_date = min(all_dates)
        # max_date = max(all_dates)
    
    def _load_data(self):
        
        # # Load attributes first to sanity-check those features before doing the compute expensive time series loading
        # self._load_combined_attributes()
        
        xr = self._load_or_create_xarray_dataset()
        
        # If is_train, split the data into train and validation periods
        if self.is_train:
            self.xr_train = xr.sel(date=slice(self.start_and_end_dates['train']['start_date'], self.start_and_end_dates['valid']['start_date']))
            self.xr_valid = xr.sel(date=slice(self.start_and_end_dates['valid']['start_date'], self.start_and_end_dates['valid']['end_date']))
        else:
            self.xr_test = xr.sel(date=slice(self.start_and_end_dates['test']['start_date'], self.start_and_end_dates['test']['end_date']))
        
        if self.cfg.loss.lower() in ['nse', 'weightednse']:
            # Get the std of the discharge for each basin, which is needed for the (weighted) NSE loss.
            self._calculate_per_basin_std(self.xr_train)
            
        if self._compute_scaler:
            # Get feature-wise center and scale values for the feature normalization
            self._setup_normalization(self.xr_train)
                        
        # Performs normalization
        # xr = (xr - self.scaler["xarray_feature_center"]) / self.scaler["xarray_feature_scale"] 
        
    def _load_or_create_xarray_dataset(self) -> xarray.Dataset:
        '''
        Load the data for all basins and create an xarray Dataset.
        
        - Returns:
            xr: xarray.Dataset, the xarray Dataset containing the data for all basins.
        '''
        
        data_list = list()
        
        # Check if static inputs are provided - it is assumed that dynamic inputs and target are always provided
        if hasattr(self.cfg, 'nn_static_inputs') and self.cfg.nn_static_inputs:
            self.cfg['nn_static_inputs'] = list()
        
        # List of columns to keep, everything else will be removed to reduce memory footprint
        keep_cols = self.cfg.nn_dynamic_inputs + self.cfg.nn_static_inputs + self.cfg.target_variables
        keep_cols = list(sorted(set(keep_cols)))
        # Lowercase all columns
        keep_cols = [col.lower() for col in keep_cols]
        
        if self.cfg.verbose:
            print("-- Loading basin data into xarray data set.")
        
        for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):
            
            df = self._load_basin_data(basin)
            
            # Make the columns to be lower case
            df.columns = [col.lower() for col in df.columns]

            # Compute mean from min-max pairs if necessary
            df = self._compute_mean_from_min_max(df, keep_cols)
            
            # Remove unnecessary columns
            df = self._remove_unnecessary_columns(df, keep_cols)
     
            # Subset the DataFrame by existing periods
            df = self._subset_df_by_periods(df)    
            
            # Convert to xarray Dataset and add basin string as additional coordinate
            xr = xarray.Dataset.from_dataframe(df.astype(self.cfg.precision['numpy']))
            xr = xr.assign_coords(basin=basin)
            data_list.append(xr)
            
        if not data_list:
            raise ValueError("No data loaded.")
        
        # Create one large dataset that has two coordinates: datetime and basin
        xr = xarray.concat(data_list, dim='basin')
            
        return xr
               
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

    def _subset_df_by_periods(self, df):  
        '''
        Subset the DataFrame by the specified periods.
        
        - Args:
            df: DataFrame, the DataFrame to subset.
        
        - Returns:
            subsetted_df: DataFrame, the subsetted DataFrame.
        '''
           
        # Initialize an empty DataFrame to store the subsetted data
        subsetted_df = pd.DataFrame()
        
        # Iterate over each interval and filter the DataFrame
        for _, dates in self.start_and_end_dates.items():
            start_date = dates['start_date']
            end_date = dates['end_date']
            interval_subset = df[(df.index >= start_date) & (df.index <= end_date)]            
            subsetted_df = pd.concat([subsetted_df, interval_subset])
            
        # Filter by unique dates
        subsetted_df = subsetted_df[~subsetted_df.index.duplicated(keep='first')]
                
        # Sort by DatetimeIndex and reindex
        subsetted_df = subsetted_df.sort_index(axis=0, ascending=True)
        # subsetted_df = subsetted_df.reindex(pd.date_range(subsetted_df.index[0], subsetted_df.index[-1], freq='D'))   ## freq=native_frequency

        return subsetted_df
        
    def _calculate_per_basin_std(self, xr: xarray.Dataset):

        basin_coordinates = xr["basin"].values.tolist()
        if self.cfg.verbose:
            print("-- Calculating target variable stds per basin")
        nan_basins = list()
        for basin in tqdm(self.basins, file=sys.stdout, disable=self._disable_pbar):
            
            obs = xr.sel(basin=basin)[self.cfg.target_variables].to_array().values
            if np.sum(~np.isnan(obs)) > 1:
                # Calculate std for each target
                per_basin_target_stds = torch.tensor(np.expand_dims(np.nanstd(obs, axis=1), 0), dtype=self.cfg.precision['torch'])
            else:
                nan_basins.append(basin)
                per_basin_target_stds = torch.full((1, obs.shape[0]), np.nan, dtype=self.cfg.precision['torch'])

            self._per_basin_target_stds[basin] = per_basin_target_stds
            
        if len(nan_basins) > 0:
            print("Warning! (Data): The following basins had not enough valid target values to calculate a standard deviation: "
                           f"{', '.join(nan_basins)}. NSE loss values for this basin will be NaN.")
            
    def _setup_normalization(self, xr: xarray.Dataset):
        '''
        Setup the normalization for the xarray dataset. 
        The default center and scale values are the feature mean and std.
        
        - Args:
            xr: xarray.Dataset, the xarray dataset to normalize.
            
        - Returns:
            None
        '''
        
        self.scaler["xarray_feature_scale"] = xr.std(skipna=True)
        self.scaler["xarray_feature_center"] = xr.mean(skipna=True)   


            
            
###############################