import os
from pathlib import Path
from typing import Dict, Tuple, Union
import xarray
import pandas as pd
import numpy as np

from src.datasetzoo.basedataset import BaseDataset
from src.utils.load_process_data import Config


class Pretrainer(BaseDataset):
    
    def __init__(self,
            cfg: Config,
            scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = dict(),
            is_train: bool = True
        ):
        
        super(Pretrainer, self).__init__(cfg, scaler, is_train)
        
    def _load_basin_dynamics(self, basin: str) -> pd.DataFrame:
        '''
        Load the basin dynamics for a basin of the pretrainer data set.
        
        - Args:
            basin: str, name of the basin.
            
        - Returns:
            df: pd.DataFrame, dataframe with the basin dynamics.
        
        '''

        df = pd.DataFrame()
        
        try:
            for period in ['train', 'valid', 'test']:
                file_path = self.cfg.data_dir / 'model_results' / f'{basin}_results_{period}.csv'
                # Check if file exists
                if not file_path.exists():
                    break
                else:
                    df_period = pd.read_csv(file_path)
                    df = pd.concat([df, df_period], axis=0)

            # Drop equal columns and sort by date
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.sort_values(by='date')

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index("date")
        except:
            pass

        return df

    def _load_attributes(self) -> pd.DataFrame:
        
        attributes_path = Path(self.cfg.concept_data_dir) / 'camels_attributes_v2.0'

        if not attributes_path.exists():
            raise RuntimeError(f"Attribute folder not found at {attributes_path}")
        
        if 'camels' in self.cfg.dataset.lower():
            txt_files = attributes_path.glob('camels_*.txt')
        else:
            raise NotImplementedError(f"No attribute files implemented for dataset {self.cfg.dataset}")
        
        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')

            dfs.append(df_temp)

        df = pd.concat(dfs, axis=1)
        # convert huc column to double digit strings
        df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
        df = df.drop('huc_02', axis=1)

        if self.basins:
            if any(b not in df.index for b in self.basins):
                raise ValueError('Some basins are missing static attributes.')
            df = df.loc[self.basins]

            # remove all attributes not defined in the config
            missing_attrs = [attr for attr in self.cfg.static_attributes if attr not in df.columns]
            if len(missing_attrs) > 0:
                raise ValueError(f'Static attributes {missing_attrs} are missing.')
            df = df[self.cfg.static_attributes]

            # Fix the order of the columns to be alphabetically
            df = df.sort_index(axis=1)

        return df



    