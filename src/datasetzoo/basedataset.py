from torch.utils.data import Dataset
from typing import Dict, Union
import xarray as xr
import pandas as pd
from tqdm import tqdm
import sys
import numpy as np
import torch
import yaml
from pathlib import Path

# Get the absolute path to the current script
script_dir = Path(__file__).resolve().parent

from src.utils.load_process_data import (
    Config, 
    load_basin_file,
)

# List of possible variables that can be passed to the model(s)
possible_variables = [
    'nn_dynamic_inputs', 
    'static_attributes', 
    'nn_mech_targets',
    'target_variables',
    'concept_inputs',
    'concept_target',
]

# Ref -> https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/datasetzoo/basedataset.py
class BaseDataset(Dataset):
    
    def __init__(self,
            cfg: Config,
            scaler: Dict[str, Union[pd.Series, xr.DataArray]] = dict(),
            is_train: bool = True
        ):
        
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        self._compute_scaler = True if is_train else False
        
        if not is_train and not scaler:
            raise ValueError("During evaluation of validation or test period, scaler dictionary has to be passed")
        self.scaler = scaler
        
        self.basins = load_basin_file(cfg.basin_file_path, self.cfg.n_first_basins, self.cfg.n_random_basins)
        
        # During training we log data processing with progress bars, but not during validation/testing
        self._disable_pbar = not self.cfg.verbose
        
        # Initialize class attributes that are filled in the data loading functions
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
        
        dataset = self._load_or_create_xarray_dataset()

        # Rename the variables back given the alias_map_clean
        dataset = dataset.rename(self.alias_map_clean)

        # If is_train, split the data into train and validation periods
        if self.is_train:
            self.ds_train = dataset.sel(date=slice(self.start_and_end_dates['train']['start_date'], self.start_and_end_dates['train']['end_date']))
            self.ds_valid = dataset.sel(date=slice(self.start_and_end_dates['valid']['start_date'], self.start_and_end_dates['valid']['end_date']))
        else:
            self.ds_test = dataset.sel(date=slice(self.start_and_end_dates['test']['start_date'], self.start_and_end_dates['test']['end_date']))
            
        if self._compute_scaler:
            # Get feature-wise center and scale values for the feature normalization
            self._setup_normalization_dynamic(self.ds_train)

        if 'static_attributes' in self.cfg._cfg and len(self.cfg.static_attributes) > 0:
            # Add static attributes to the DataFrame
            static_df = self._load_attributes()

            if self._compute_scaler:
                self.ds_static = self._setup_normalization_static(static_df)
        else:
            self.ds_static = None

    def _load_or_create_xarray_dataset(self) -> xr.Dataset:
        '''
        Load the data for all basins and create an xarray Dataset.
        
        - Returns:
            ds: xarray.Dataset, the xarray Dataset containing the data for all basins.
        '''
        
        data_list = list()
        
        # List of columns to keep, everything else will be removed to reduce memory footprint
        # Create keep_cols list dynamically
        keep_cols = []
        for attr in possible_variables:
            if attr in self.cfg._cfg:
                keep_cols.extend(getattr(self.cfg, attr))
        
        keep_cols = list(sorted(set(keep_cols)))
        # Lowercase all columns
        keep_cols = [col.lower() for col in keep_cols]

        # Get the alias map
        self.alias_map = self._get_alias_map(keep_cols)
        
        # if self.cfg.verbose:
        print("-- Loading basin dynamics into xarray data set.")
        
        for basin in tqdm(self.basins, disable=self._disable_pbar, file=sys.stdout):
            
            df = self._load_basin_dynamics(basin)

            if df.empty:
                print(f"Warning! (Data): No data loaded for basin {basin}. Skipping.")
                continue
            
            # Make the columns to be lower case
            df.columns = [col.lower() for col in df.columns]

            # Compute mean from min-max pairs if necessary
            df = self._compute_mean_from_min_max(df, keep_cols)

            # Remove unnecessary columns
            df = self._remove_unnecessary_columns(df, keep_cols)

            # Subset the DataFrame by existing periods
            df = self._subset_df_by_periods(df)   
            
            # Convert to xarray Dataset and add basin string as additional coordinate
            ds = xr.Dataset.from_dataframe(df.astype(self.cfg.precision['numpy']))
            ds = ds.assign_coords(basin=basin)
            data_list.append(ds)
            
        if not data_list:
            raise ValueError("No data loaded.")
        
        # Create one large dataset that has two coordinates: datetime and basin
        ds = xr.concat(data_list, dim='basin')
            
        return ds
               
    def _load_basin_dynamics(self, basin: str) -> pd.DataFrame:
        """This function has to return the data for the specified basin as a time-indexed pandas DataFrame"""
        raise NotImplementedError("This function has to be implemented by the child class")
    
    def _load_attributes(self) -> pd.DataFrame:
        """This function has to return the static attributes for the specified basins as a pandas DataFrame"""
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

        if any(col.startswith('tmean') for col in keep_cols) and \
            not any(col.startswith('tmean') for col in df.columns):

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
                    
        return df

    @staticmethod
    def _get_alias_map(keep_cols):
        '''
        Get the alias map for the specified columns.
        
        - Args:
            keep_cols: list, a list of columns to keep.
            
        - Returns:
            alias_map: dict, a dictionary mapping the original column names to the aliases.
        '''

        # Construct the path to the 'variable_aliases.yml' file
        variable_aliases_path = script_dir / '..' / 'utils' / 'variable_aliases.yml'

        # Open and read the 'variable_aliases.yml' file
        with open(variable_aliases_path, 'r') as file:
            aliases = yaml.safe_load(file)

        alias_map = {}
        for col in keep_cols:
            if col in aliases:
                alias_map[col] = aliases[col]
                    
        return alias_map

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
        updated_alias_map = {}
        not_available_columns  = []

        # print(df.head())
        # aux = input('Press any key to continue...')
        
        for key, aliases in self.alias_map.items():
            # Find the intersection of DataFrame columns and aliases
            matched_columns = [alias for alias in aliases if alias in df.columns]
            
            if matched_columns:
                # Update the alias map with only the matched columns
                updated_alias_map[key] = matched_columns[0]  # This assumes that there is only one matched column
            else:
                # Record the key if no matched column is found
                not_available_columns.append(key)

        if not_available_columns:
            msg = [
                f"The following features are not available in the data: {not_available_columns}. ",
                f"These are the available features: {df.columns.tolist()}. ",
                "Check the 'variable_aliases.yml' file for the correct column names."
            ]
            raise KeyError("".join(msg))
        
        # Create the alias map with the matched columns (inverted)
        self.alias_map_clean = {v: k for k, v in updated_alias_map.items()}

        # Filter the DataFrame to keep only the matched columns
        df = df[list(updated_alias_map.values())]
        
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
            
    def _setup_normalization_dynamic(self, ds: xr.Dataset):
        '''
        Setup the normalization for the xarray dataset. 
        The default center and scale values are the feature mean and std.
        
        - Args:
            ds: xarray.Dataset, the xarray dataset to normalize.
            
        - Returns:
            None
        '''
        
        self.scaler["ds_feature_std"] = ds.groupby('basin').std(dim='date')
        self.scaler["ds_feature_mean"] = ds.groupby('basin').mean(dim='date')
        # self.scaler["ds_feature_min"] = ds.groupby('basin').min(dim='date')

        # print("ds_feature_mean:", self.scaler["ds_feature_mean"])
        # aux = input('Press any key to continue...')
    
    def _setup_normalization_static(self, df: pd.DataFrame):
        '''
        Setup the normalization for the static attributes. 
        The default center and scale values are the feature mean and std.
        
        - Args:
            df: pd.DataFrame, the DataFrame to normalize.
            
        - Returns:
            None
        '''
        
        self.scaler["static_std"] = df.std()
        self.scaler["static_mean"] = df.mean()

        # Normalize the static attributes
        df = (df - self.scaler["static_mean"]) / self.scaler["static_std"]

        # Convert to xarray Dataset
        ds_static = xr.Dataset.from_dataframe(df)

        return ds_static


###############################