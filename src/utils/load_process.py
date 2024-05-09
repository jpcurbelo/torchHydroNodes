import yaml
import os
from datetime import datetime
import pandas as pd
from pandas import Timestamp
from pathlib import Path
from typing import List, Union, Any
import numpy as np
import torch

## Functions      
def load_basin_file(basin_file: Path) -> List[str]:
    '''
    Load the basin file and return the list of basins.
    
    - Args:
        basin_file: str, path to the basin file.
        
    - Returns:
        basins: list, list of basins.
    '''
        
    with open(basin_file, 'r') as f:
        basins = f.read().splitlines()
    
    # Convert basins to strings of 8 characters with leading zeros if needed
    basins = [f"{int(basin):08d}" if basin.isdigit() else basin for basin in basins]

    return basins    
                            
def load_forcing_target_data(run_config):
    
    pass

## Classes
class Config(object):
    '''
    During parsing, config keys that contain 'dir', 'file', or 'path' will be converted to pathlib.Path instances.
    '''
    
    def __init__(self, yml_path_or_dict: Union[Path, dict]):
        
        if isinstance(yml_path_or_dict, Path) or isinstance(yml_path_or_dict, str):
            self._cfg = self._parse_run_config(yml_path_or_dict)
        else:
            raise ValueError(f'Cannot create a config from input of type {type(yml_path_or_dict)}.')
        
        # Create a folder to save the trained models
        if isinstance(yml_path_or_dict, Path):
            self.create_run_folder_tree()
            
        # Dump the configuration data to a ymal file
        self.dump_config()
        
    def create_run_folder_tree(self):
        '''
        Create a folder to save the trained models.
        
        - Args:
            config_data: dict, configuration data.
            
        - Returns:
            None
        '''
        
        config_dir = self._cfg['config_dir']  
        if not os.path.exists(config_dir / 'runs'):
            os.mkdir(config_dir / 'runs')
            
        # Experiment name
        if "experiment_name" in self._cfg and self._cfg["experiment_name"]:
            experiment_name = self._cfg['experiment_name']
        else:
            experiment_name = 'run'
                               
        # Create a folder to save the trained models
        now = datetime.now()
        dt_string = now.strftime("%y%m%d %H%M%S")
        run_dir = config_dir / 'runs' / f'{experiment_name}_{dt_string.split()[0]}_{dt_string.split()[1]}'
        try:
            os.mkdir(run_dir)
            self._cfg['run_dir'] = run_dir
        except OSError as error:
            print(f"Folder '{run_dir}' already existed.")
            
        # Create a folder to save img content
        plots_dir = os.path.join(run_dir, 'model_plots')
        os.mkdir(plots_dir)
        self._cfg['plots_dir'] = plots_dir
        
        # Create a folder to save the model results (csv files)
        results_dir = os.path.join(run_dir, 'model_results')
        os.mkdir(results_dir)
        self._cfg['results_dir'] = results_dir
            
        # Check if forcings are provided for camelus dataset
        # if config_data['dataset'] == 'camelsus' and ('forcings' not in config_data or not config_data['forcings']):
        #     config_data['forcings'] = ['daymet']
        if self.dataset == 'camelsus' and not self.forcings:
            self._cfg['forcings'] = ['daymet']
            # Raise a warning
            print('Warning! (Data): Forcing data not provided. Using Daymet data as default.')

        # Convert PosixPath objects to strings before serializing
        cfg_copy = self._cfg.copy()
        for key, value in cfg_copy.items():
            if isinstance(value, Path):
                cfg_copy[key] = str(value)
            elif isinstance(value, Timestamp):
                cfg_copy[key] = value.isoformat()
               
        # Convert precision to string
        if cfg_copy['precision']['numpy'] == np.float32:
            cfg_copy['precision'] = 'float32'
        else:
            cfg_copy['precision'] = 'float64'
    
    def dump_config(self, filename: str = 'config.yml'):
        '''
        Dump the configuration data to a ymal file.
        
        - Args:
            None
            
        - Returns:
            None
        '''
        
        # Convert PosixPath objects to strings before serializing
        cfg_copy = self._cfg.copy()
        for key, value in cfg_copy.items():
            if isinstance(value, Path):
                cfg_copy[key] = str(value)
            elif isinstance(value, Timestamp):
                cfg_copy[key] = value.isoformat()
                
        # Convert precision to string
        if cfg_copy['precision']['numpy'] == np.float32:
            cfg_copy['precision'] = 'float32'
        else:
            cfg_copy['precision'] = 'float64'
            
        # Save the configuration data to a ymal file
        config_path = self._cfg['run_dir'] / filename
        with open(config_path, 'w') as f:
            yaml.dump(cfg_copy, f)
    
    def _get_property_value(self, key: str) -> Union[float, int, str, list, dict, Path, pd.Timestamp]:
        '''
        Get the value of a property from the config.
        
        - Args:
            key: str, key of the property.
            
        - Returns:
            value: float, int, str, list, dict, Path, pd.Timestamp, value of the property.
        '''
        
        """Use this function internally to return attributes of the config that are mandatory"""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return self._cfg[key]    
        
    @staticmethod
    def _parse_run_config(config_file):
        '''
        Parse the configuration data from a YAML file or a dictionary.
        
        - Args:
            config_file: str, path to the YAML file with the configuration data.
            
        - Returns:
            cfg: dict, configuration data.
        '''
        
        # Read the config file
        if isinstance(config_file, Path):
            if config_file.exists():
                with open(config_file, 'r') as ymlfile:
                    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)                    
            else:
                raise FileNotFoundError(f"File not found: {config_file}")
        else:
            cfg = config_file
            
        # Extract parent directory of the config file
        cfg['config_dir'] = config_file.parent
        
        # Define basin file directory
        if 'basin_file_path' not in cfg:
            cfg['basin_file_path'] = os.path.join(cfg['config_dir'], cfg['basin_file'])
         
        # Parse the config dictionary
        for key, val in cfg.items():
            # Convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = list()
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None
                    
            # Convert all date strings to datetime objects
            if key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = list()
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d/%m/%Y'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')
                    
        # Set precision
        if 'precision' not in cfg or cfg['precision'] not in ['float32', 'float64']:
            cfg['precision'] = {
                'numpy': np.float32,
                'torch': torch.float32
            }
        else:
            cfg['precision'] = {
                'numpy': np.float32 if cfg['precision'] == 'float32' else np.float64,
                'torch': torch.float32 if cfg['precision'] == 'float32' else torch.float64
            }
            
        # Add more config parsing if necessary
        return cfg
         
    @staticmethod
    def _as_default_list(value: Any) -> list:
        '''
        Convert the value to a list if it is not a list.
        
        - Args:
            value: Any, value to convert to a list.
            
        - Returns:
            list, value as a list.
        '''
        
        if value is None:
            return list()
        elif isinstance(value, list):
            return value
        else:
            return [value]      
        
    @property
    def dataset(self) -> str:
        return self._get_property_value("dataset")
    
    @property
    def basin_file(self) -> Path:
        return self._get_property_value("basin_file")
    
    @property
    def basin_file_path(self) -> Path:
        return self._get_property_value("basin_file_path")
    
    @property
    def nn_dynamic_inputs(self) -> list:
        return self._get_property_value("nn_dynamic_inputs")
    
    @property
    def nn_static_inputs(self) -> list:
        if "static_attributes" in self._cfg.keys():
            self._as_default_list(self._cfg['nn_static_inputs'])
        else:
            return list()
    
    @property
    def target_variables(self) -> list:
        return self._get_property_value("target_variables")
    
    @property
    def forcings(self) -> List[str]:
        return self._as_default_list(self._get_property_value("forcings"))
        
    @property
    def data_dir(self) -> Path:
        return self._get_property_value("data_dir")

    @property
    def precision(self) -> dict:
        return self._get_property_value("precision")
    
    def device(self) -> torch.device:
        return self._get_property_value("device")

    @property
    def loss(self) -> str:
        return self._get_property_value("loss")
    
    @property
    def verbose(self) -> int:
        """Defines level of verbosity.

        0: Only log info messages, don't show progress bars
        1: Log info messages and show progress bars

        Returns
        -------
        int
            Level of verbosity.
        """
        return self._cfg.get("verbose", 1)

    @property
    def train_start_date(self) -> pd.Timestamp:
        return self._get_property_value("train_start_date")
    
    @property
    def train_end_date(self) -> pd.Timestamp:
        return self._get_property_value("train_end_date")
    
    @property
    def test_start_date(self) -> pd.Timestamp:
        return self._get_property_value("test_start_date")
    
    @property
    def test_end_date(self) -> pd.Timestamp:
        return self._get_property_value("test_end_date")
    
    @property
    def valid_start_date(self) -> pd.Timestamp:
        return self._get_property_value("valid_start_date")
    
    @property
    def valid_end_date(self) -> pd.Timestamp:
        return self._get_property_value("valid_end_date")
    
    @property   
    def concept_model(self) -> str:
        return self._get_property_value("concept_model")
    
    @property
    def nn_model(self) -> str:
        return self._get_property_value("nn_model")
    
    @property
    def hidden_layers(self) -> List[int]:
        return self._get_property_value("hidden_layers")

    @property
    def run_dir(self) -> Path:
        return self._get_property_value("run_dir")
    
    @property
    def plots_dir(self) -> Path:
        return self._get_property_value("plots_dir")
    
    @property
    def results_dir(self) -> Path:
        return self._get_property_value("results_dir")
    
    @property
    def disable_pbar(self) -> bool:
        return not self._cfg.get("verbose")
    
    @property
    def metrics(self) -> List[str]:
        return self._as_default_list(self._get_property_value("metrics"))
    

if __name__ == "__main__":
    pass