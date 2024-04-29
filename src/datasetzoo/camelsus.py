import os
from pathlib import Path
from typing import Dict, Tuple, Union
import xarray
import pandas as pd
import numpy as np

from src.datasetzoo.basedataset import BaseDataset


class CamelsUS(BaseDataset):
    
    def __init__(self,
            cfg: dict,
            is_train: bool = True,
            period: str = "train",
            basin: str = None,
            scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        
        super(CamelsUS, self).__init__(cfg, is_train, period, scaler)
        
    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        # get forcings
        dfs = []
        for forcing in self.cfg.forcings:
            df, area = load_camels_us_forcings(self.cfg.data_dir, basin, forcing)

            # rename columns
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        
        # add discharge
        df['obs_runoff(mm/day)'] = load_camels_us_discharge(self.cfg.data_dir, basin, area)   

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        return df
    
    # def _load_attributes(self) -> pd.DataFrame:
    #     return load_camels_us_attributes(self.cfg.data_dir, basins=self.basins)
    
    
def load_camels_us_forcings(data_dir: Path, basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:
    '''
    Load the forcing data for a basin of the CAMELS US data set.
    
    - Args:
        data_dir: Path, path to the CAMELS US directory. This folder must contain a 'basin_mean_forcing' folder with 18
                  subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the forcing files
                    (.txt), starting with the 8-digit basin id.
        basin: str, 8-digit USGS identifier of the basin.
        forcings: str, name of the forcing data to load.
        
    - Returns:
        df: pd.DataFrame, time-index pandas.DataFrame of the forcing values.
        area: int, catchment area (m2), used to normalize the discharge.
    '''
    
    forcing_path = data_dir / 'basin_mean_forcing' / forcings
    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")
    
    file_path = list(forcing_path.glob(f'**/{basin}_*_forcing_leap.txt'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}/**')
    
    with open(file_path, 'r') as fp:
        # load area from header
        fp.readline()
        fp.readline()
        area = int(fp.readline())
        # load the dataframe from the rest of the stream
        df = pd.read_csv(fp, sep=r'\s+')
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str),
                                    format="%Y/%m/%d")
        df = df.set_index("date")

    return df, area

def load_camels_us_discharge(data_dir: Path, basin: str, area: int) -> pd.Series:
    """Load the discharge data for a basin of the CAMELS US data set.

    Parameters
    ----------
    data_dir : Path
        Path to the CAMELS US directory. This folder must contain a 'usgs_streamflow' folder with 18
        subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the discharge files 
        (.txt), starting with the 8-digit basin id.
    basin : str
        8-digit USGS identifier of the basin.
    area : int
        Catchment area (m2), used to normalize the discharge.

    Returns
    -------
    pd.Series
        Time-index pandas.Series of the discharge values (mm/day)
    """

    discharge_path = data_dir / 'usgs_streamflow'
    
    file_path = list(discharge_path.glob(f'**/{basin}_streamflow_qc.txt'))
    if file_path:
        file_path = file_path[0]
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str), format="%Y/%m/%d")
    df = df.set_index("date")

    # normalize discharge from cubic feet per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

    return df.QObs