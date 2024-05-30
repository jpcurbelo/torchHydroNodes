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
        
    def _load_basin_data(self, basin: str) -> pd.DataFrame:

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



    