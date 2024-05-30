import yaml
import os
from datetime import datetime
import pandas as pd
from pandas import Timestamp
from pathlib import Path
from typing import List, Union, Any
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset

# Get the absolute path to the current script
script_dir = Path(__file__).resolve().parent

## Functions      
def load_basin_file(basin_file: Path, n_first_basins: int = -1, n_random_basins: int = -1) -> list:
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
    if n_first_basins > 0:
        basins = [f"{int(basin):08d}" if basin.isdigit() else basin for basin in basins][:n_first_basins]
    else:
        basins = [f"{int(basin):08d}" if basin.isdigit() else basin for basin in basins]

    # Select a random subset of basins
    if n_random_basins > 0:
        basins = np.random.choice(basins, n_random_basins, replace=False)

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

        if 'run_dir' not in self._cfg:
        
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
        if 'dataset' in self._cfg and self._cfg['dataset'] == 'camelsus' and 'forcings' not in self._cfg:
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
                # cfg_copy[key] = value.isoformat()
                # To string this format 01/10/1980 (DD/MM/YYYY)
                cfg_copy[key] = value.strftime('%d/%m/%Y')
                
        # Convert precision to string
        if cfg_copy['precision']['numpy'] == np.float32:
            cfg_copy['precision'] = 'float32'
        else:
            cfg_copy['precision'] = 'float64'
            
        # Device to string
        cfg_copy['device'] = str(cfg_copy['device'])
        
        # Save the configuration data to a ymal file
        config_path = self._cfg['run_dir'] / filename
        with open(config_path, 'w') as f:
            yaml.dump(cfg_copy, f)
    
    def _get_property_value(self, key: str, default=None) -> Union[float, int, str, list, dict, Path, pd.Timestamp, torch.device]:
        '''
        Get the value of a property from the config.

        - Args:
            key: str, key of the property.
            default: value to return if key is not found.
            
        - Returns:
            value: float, int, str, list, dict, Path, pd.Timestamp, value of the property.
        '''
        if key not in self._cfg.keys():
            if default is not None:
                return default
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return self._cfg[key] 
        
    def _parse_run_config(self, config_file):
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

        # Set device (GPU or CPU)
        try:
            gpu = int(cfg['device'].split(':')[-1])
        except:
            gpu = None

        if 'device' not in cfg or cfg['device'] is None:
            cfg['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif gpu is not None and gpu >= 0:
            cfg['device'] = torch.device(f"cuda:{gpu}")
        else:
            cfg['device'] = torch.device("cpu")
        
        # Load variables for the concept model
        if 'concept_model' in cfg:
            cfg['concept_inputs'],  cfg['concept_target'] = self._load_concept_model_vars(cfg['concept_model'])
       
        # Get periods from config file
        periods = []
        for period in ['train', 'valid', 'test']:
            start_key = f"{period}_start_date"
            end_key = f"{period}_end_date"
            if start_key in cfg and end_key in cfg:
                periods.append(period)

        cfg['periods'] = periods

        # Check log_n_basins and convert to a list of str with 8 characters and leading zeros if necessary
        if 'log_n_basins' in cfg and isinstance(cfg['log_n_basins'], list):
            # Convert log_n_basins to a list of str with 8 characters and leading zeros
            cfg['log_n_basins'] = [f"{int(basin):08d}" if isinstance(basin, int) else basin for basin in cfg['log_n_basins']]

        # Add more config parsing if necessary
        return cfg
    
    @staticmethod
    def _load_concept_model_vars(concept_model: str) -> dict:

        # Load utils/concept_model_vars.yml
        with open(script_dir / 'concept_model_vars.yml', 'r') as f:
            var_alias = yaml.load(f, Loader=yaml.FullLoader)

        # Load the variables for the concept model
        var_inputs = var_alias[concept_model]['model_inputs']
        var_outputs = var_alias[concept_model]['model_target']

        return var_inputs, var_outputs
         
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
    def nn_mech_targets(self) -> list:
        return self._get_property_value("nn_mech_targets")

    @property
    def target_variables(self) -> list:
        return self._get_property_value("target_variables")
    
    @property
    def concept_inputs(self) -> list:
        return self._get_property_value("concept_inputs")
    
    @property
    def concept_target(self) -> list:
        return self._get_property_value("concept_target")

    @property
    def forcings(self) -> List[str]:
        return self._as_default_list(self._get_property_value("forcings"))
        
    @property
    def data_dir(self) -> Path:
        return self._get_property_value("data_dir")

    @data_dir.setter
    def data_dir(self, value: Path):
        self._cfg['data_dir'] = value

    @property
    def precision(self) -> dict:
        return self._get_property_value("precision", default={'numpy': np.float32, 'torch': torch.float32})
    
    @property
    def device(self) -> torch.device:
        return self._get_property_value("device", default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    @device.setter
    def device(self, value: torch.device):
        self._device = value

    @property
    def loss(self) -> str:
        return self._get_property_value("loss", default="MSE")
    
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
        # return self._cfg.get("verbose", 1)
        return self._get_property_value("verbose", default=1)

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
        return self._get_property_value("nn_model", default="mlp")
    
    @property
    def hidden_size(self) -> List[int]:
        return self._get_property_value("hidden_size")

    @property
    def run_dir(self) -> Path:
        return self._get_property_value("run_dir")
    
    @property
    def config_dir(self) -> Path:
        return self._get_property_value("config_dir")

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
    
    @property
    def batch_size(self) -> int:
        return self._get_property_value("batch_size", default=256)

    @property
    def num_workers(self) -> int:
        return self._get_property_value("num_workers", default=8)
    
    @property
    def epochs(self) -> int:
        return self._get_property_value("epochs", default=10)
    
    @property
    def optimizer(self) -> str:
        return self._get_property_value("optimizer", default="Adam")
    
    @property
    def learning_rate(self) -> float:
        return self._get_property_value("learning_rate", default=0.001)





    @property
    def log_n_basins(self) -> int:
        return self._get_property_value("log_n_basins", default=0)

    @property
    def log_every_n_epochs(self) -> int:
        return self._get_property_value("log_every_n_epochs", default=10)

    @property
    def periods(self) -> List[str]:
        return self._get_property_value("periods")

    @property
    def seed(self) -> int:
        return self._get_property_value("seed", default=1111)

    @property
    def n_first_basins(self) -> int:
        return self._get_property_value("n_first_basins", default=-1)
    
    @property
    def n_random_basins(self) -> int:
        return self._get_property_value("n_random_basins", default=-1)

    @property
    def clip_gradient_norm(self) -> float:
        return self._get_property_value("clip_gradient_norm", default=1.0)

    @clip_gradient_norm.setter
    def clip_gradient_norm(self, value: float):
        self._cfg['clip_gradient_norm'] = value

    @property
    def metrics(self) -> List[str]:
        return self._as_default_list(self._get_property_value("metrics"))
    
## Classes for data loading
class BatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, shuffle=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_batches = (dataset_len + batch_size - 1) // batch_size

    def __iter__(self):
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(indices)
        batches = [indices[i * self.batch_size:(i + 1) * self.batch_size] for i in range(self.num_batches)]
        return iter(batches)

    def __len__(self):
        return self.num_batches

class CustomDatasetToNN(Dataset):
    def __init__(self, input_tensor, output_tensor, basin_ids):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor
        self.basin_ids = basin_ids

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.output_tensor[idx], self.basin_ids[idx]

class BasinBatchSampler(Sampler):
    def __init__(self, basin_ids, batch_size, shuffle=True):
        self.basin_ids = basin_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create a mapping from basin_id to indices
        self.basin_to_indices = {}
        for idx, basin in enumerate(basin_ids):
            if basin not in self.basin_to_indices:
                self.basin_to_indices[basin] = []
            self.basin_to_indices[basin].append(idx)
        
        # Generate batches
        self.batches = self._create_batches()

    def _create_batches(self):
        batches = []
        for _, indices in self.basin_to_indices.items():
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


if __name__ == "__main__":
    pass