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
import random
from scipy.interpolate import Akima1DInterpolator
import xarray as xr
import torch.nn as nn

# Get the absolute path to the current script
script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent.parent

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

def update_hybrid_cfg(cfg):

    # Load vars from the nn_model
    nn_cgf_dir = project_dir / 'data' / cfg.nn_model_dir / 'config.yml'
    
    # Load the nn_model config file
    if nn_cgf_dir.exists():
        with open(nn_cgf_dir, 'r') as ymlfile:
            cfg_nn = yaml.load(ymlfile, Loader=yaml.FullLoader)
    else:
        raise FileNotFoundError(f"File not found: {nn_cgf_dir}")
    
    cfg.nn_dynamic_inputs = cfg_nn['nn_dynamic_inputs']
    cfg.hidden_size = cfg_nn['hidden_size']
    cfg.nn_mech_targets = cfg_nn['nn_mech_targets']
    cfg.target_variables = cfg_nn['target_variables']


    return cfg


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

        # Set seed for reproducibility
        self.set_random_seeds()

        # Create a folder to save the trained models and dump the configuration data to a ymal file
        if 'run_dir' not in self._cfg:
        
            # Create a folder to save the trained models
            if isinstance(yml_path_or_dict, Path):
                self.create_run_folder_tree()
                
            # Dump the configuration data to a ymal file
            self.dump_config()

    def set_random_seeds(self):
        '''
        Set random seeds for reproducibility
        '''
        if 'seed' not in self._cfg or self._cfg['seed'] is None:
            seed = int(np.random.uniform(low=0, high=1e6))
            self._cfg['seed'] = seed
        else:
            seed = self._cfg['seed']

        # fix random seeds for various packages
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        
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

        # Number of basins
        n_first_basins = cfg_copy['n_first_basins'] if 'n_first_basins' in cfg_copy else -1
        n_random_basins = cfg_copy['n_random_basins'] if 'n_random_basins' in cfg_copy else -1
        cfg_copy['number_of_basins'] = len(load_basin_file(cfg_copy['basin_file_path'], n_first_basins, n_random_basins))
        self._cfg['number_of_basins'] = cfg_copy['number_of_basins']

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
    
    @nn_dynamic_inputs.setter
    def nn_dynamic_inputs(self, value: list):
        self._cfg['nn_dynamic_inputs'] = value
        
    @property
    def nn_static_inputs(self) -> list:
        if "static_attributes" in self._cfg.keys():
            self._as_default_list(self._cfg['nn_static_inputs'])
        else:
            return list()
    
    @property
    def nn_mech_targets(self) -> list:
        return self._get_property_value("nn_mech_targets")

    @nn_mech_targets.setter
    def nn_mech_targets(self, value: list):
        self._cfg['nn_mech_targets'] = value

    @property
    def scale_target_vars(self) -> bool:
        return self._get_property_value("scale_target_vars", default=False)

    @property
    def target_variables(self) -> list:
        return self._get_property_value("target_variables")
    
    @target_variables.setter
    def target_variables(self, value: list):
        self._cfg['target_variables'] = value

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
    def nn_model_dir(self) -> Path:
        return self._get_property_value("nn_model_dir", default=None)

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
    def hybrid_model(self) -> str:
        return self._get_property_value("hybrid_model", default="exphydroM100")

    @property
    def hidden_size(self) -> List[int]:
        return self._get_property_value("hidden_size")

    @hidden_size.setter
    def hidden_size(self, value: List[int]):
        self._cfg['hidden_size'] = value

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
        return self._get_property_value("verbose", default=False) == False
    
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
    def patience(self) -> int:
        return self._get_property_value("patience", default=10)

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
    def number_of_basins(self) -> int:
        return self._get_property_value("number_of_basins")

    @property
    def clip_gradient_norm(self) -> float:
        return self._get_property_value("clip_gradient_norm", default=1.0)

    @clip_gradient_norm.setter
    def clip_gradient_norm(self, value: float):
        self._cfg['clip_gradient_norm'] = value

    @property
    def metrics(self) -> List[str]:
        return self._as_default_list(self._get_property_value("metrics"))

    @property
    def odesmethod(self) -> str:
        return self._get_property_value("odesmethod", default="RK23")

## Classes for data loading
class BatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, shuffle=False):
        self.dataset_len = dataset_len
        if batch_size == -1:
            batch_size = dataset_len
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
    def __init__(self, basin_ids, batch_size, shuffle=False):
        self.basin_ids = basin_ids
        if batch_size == -1:
            batch_size = len(basin_ids)
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

class ExpHydroCommon:
    
    def create_interpolator_dict(self, is_trainer=False):
        '''
        Create interpolator functions for the input variables.
        
        - Returns:
            interpolators: dict, dictionary with the interpolator functions for each basin and variable.
        '''
        
        if is_trainer:
            # Concatenate self. pretrainer.fulldataset.ds_train and self. pretrainer.fulldataset.ds_valid
            dataset = xr.concat([self.pretrainer.fulldataset.ds_train, self.pretrainer.fulldataset.ds_valid], dim='date')
            time_series = np.linspace(0, len(dataset['date'].values) - 1, len(dataset['date'].values))
        else:
            dataset = self.dataset
            time_series = self.time_series

        # Create a dictionary to store interpolator functions for each basin and variable
        interpolators = dict()
        
        # Loop over the basins and variables
        for basin in dataset['basin'].values:
    
            interpolators[basin] = dict()
            for var in self.interpolator_vars:
                                
                # Get the variable values
                var_values = dataset[var].sel(basin=basin).values

                # Interpolate the variable values
                interpolators[basin][var] = Akima1DInterpolator(time_series, var_values)
                
        return interpolators
    
    def get_parameters(self):
        '''
        Get the parameters for the model from the parameter file.
        
        - Returns:
            params_dict: dict, dictionary with the parameters for each basin.
            
        '''
        
        params_dir = Path(__file__).resolve().parent.parent / 'modelzoo_concept' \
            / 'bucket_parameter_files' / f'bucket_{self.cfg.concept_model}.csv'
        
        try:
            params_df = pd.read_csv(params_dir)

        except FileNotFoundError:
            raise FileNotFoundError(f"File '{params_dir}' not found. Check the file path.")
        else:
            
            # Remove UNKNOWN column if it exists
            if 'UNKNOWN' in params_df.columns:
                params_df = params_df.drop(columns=['UNKNOWN'])
                
            # Make basinID to be integer if it is not
            if params_df['basinID'].dtype == 'float':
                params_df['basinID'] = params_df['basinID'].astype(int)
                
            params_dict = dict()
            # Loop over the basins and extract the parameters
            for basin in self.dataset['basin'].values:
                
                # Convert basin to int to match the parameter file
                basin_int = int(basin)
                    
                try:
                    params_opt = params_df[params_df['basinID'] == basin_int].values[0]
                except IndexError:
                    # Raise warning but continue
                    # raise ValueError(f"Basin {basin} not found in the parameter file.")
                    print(f"Warning! (Data): Basin {basin} not found in the parameter file.")
                    
                # S0,S1,f,Smax,Qmax,Df,Tmax,Tmin
                params_dict[basin] = params_opt[1:]
      
            return params_dict

    def scale_target_vars(self, is_trainer=False):
       
        epsilon = 1e-16  # Small constant to avoid division by zero and log of zero

        # print(self.dataset)

        # print('Scaling target variables...')

        # Scale the target variables
        if is_trainer:
            # print('Scaling target variables for the trainer model...')
            q_values = self.dataset['obs_runoff'].values
            # self.dataset['obs_runoff'] = (('basin', 'date'), np.log(np.where(q_values == 0, epsilon, q_values)))  # Avoid log(0)
            self.dataset['obs_runoff'] = (('basin', 'date'), np.log(q_values)) 
            # print("self.dataset['obs_runoff']", self.dataset['obs_runoff'])
        else:
            # print('Scaling input variables for the pretrainer model...')
            ## Psnow -> arcsinh
            self.dataset['ps_bucket'] = (('basin', 'date'), np.arcsinh(self.dataset['ps_bucket'].values))
            ## Prain -> arcsinh
            self.dataset['pr_bucket'] = (('basin', 'date'), np.arcsinh(self.dataset['pr_bucket'].values))
            ## M -> arcsinh
            self.dataset['m_bucket'] = (('basin', 'date'), np.arcsinh(self.dataset['m_bucket'].values))
            ## ET -> log(ET / dayl)
            self.dataset['et_bucket'] = (('basin', 'date'), np.log(self.dataset['et_bucket'].values / self.dataset['dayl'].values))
            ## Q -> log(Q)
            q_values = self.dataset['q_bucket'].values
            # self.dataset['q_bucket'] = (('basin', 'date'), np.log(np.where(q_values == 0, epsilon, q_values)))  # Avoid log(0)
            self.dataset['q_bucket'] = (('basin', 'date'), np.log(q_values))  

        # return self.dataset
        # # return tensor_dict

    def scale_back_simulated(self, outputs, ds_basin, is_trainer=False):
        # Transfer outputs to CPU if necessary
        if outputs.is_cuda:
            outputs = outputs.cpu()

        # Scale back the output variables
        if is_trainer:
            outputs = torch.exp(outputs)
        else:
            # Psnow -> arcsinh
            outputs[:, 0] = torch.sinh(outputs[:, 0])
            # Prain -> arcsinh
            outputs[:, 1] = torch.sinh(outputs[:, 1])
            # M -> arcsinh
            outputs[:, 2] = torch.sinh(outputs[:, 2])
            # ET -> log(ET / dayl)
            outputs[:, 3] = torch.exp(outputs[:, 3]) * torch.tensor(ds_basin.dayl.values, dtype=torch.float32)
            # Q -> log(Q)
            outputs[:, 4] = torch.exp(outputs[:, 4])   

        return outputs

    @staticmethod
    def scale_back_observed(ds_basin, is_trainer=False):

        # Psnow -> arcsinh
        ps_values = np.sinh(ds_basin['ps_bucket'].values)
        ds_basin['ps_bucket'] = xr.DataArray(ps_values, dims=['date'])

        # Prain -> arcsinh
        pr_values = np.sinh(ds_basin['pr_bucket'].values)
        ds_basin['pr_bucket'] = xr.DataArray(pr_values, dims=['date'])

        # M -> arcsinh
        m_values = np.sinh(ds_basin['m_bucket'].values)
        ds_basin['m_bucket'] = xr.DataArray(m_values, dims=['date'])

        # ET -> log(ET / dayl)
        et_values = ds_basin['et_bucket'].values
        dayl_values = ds_basin['dayl'].values
        et_bucket_values = np.exp(et_values) * dayl_values  # Scale back using exp and dayl
        ds_basin['et_bucket'] = xr.DataArray(et_bucket_values, dims=['date'])
        
        # Q -> log(Q)
        if is_trainer:
            q_bucket_values = np.exp(ds_basin['obs_runoff'].values)
            ds_basin['obs_runoff'] = xr.DataArray(q_bucket_values, dims=['date'])
        else:
            q_bucket_values = np.exp(ds_basin['q_bucket'].values)
            ds_basin['q_bucket'] = xr.DataArray(q_bucket_values, dims=['date'])

        return ds_basin

    @property
    def interpolator_vars(self):
        return ['prcp', 'tmean', 'dayl']


class ExpHydroODEs(nn.Module):
    def __init__(self, 
                # precp_series,
                # tmean_series,
                # lday_series,
                # time_series,

                precp_interp,
                temp_interp,
                lday_interp,
                data_type_torch,
                device,

                scale_target_vars,
                pretrainer,
                basin,
                step_function):
        super(ExpHydroODEs, self).__init__()

        # self.precp_series = precp_series
        # self.tmean_series = tmean_series
        # self.lday_series = lday_series
        # self.time_series = time_series

        self.precp_interp = precp_interp
        self.temp_interp = temp_interp
        self.lday_interp = lday_interp
        self.data_type_torch = data_type_torch
        self.device = device

        self.scale_target_vars = scale_target_vars
        self.pretrainer = pretrainer
        self.basin = basin
        self.step_function = step_function

    def forward(self, t, y):
         
        ## Unpack the state variables
        # S0: Storage state S_snow (t)                       
        # S1: Storage state S_water (t)   
        s0 = y[..., 0] #.to(self.device)
        s1 = y[..., 1] #.to(self.device)

        # # Find left index for the interpolation
        # idx = torch.searchsorted(self.time_series, t, side='right') - 1
        # idx = idx.clamp(max=self.time_series.size(0) - 2)  # Ensure indices do not exceed valid range

        # # Linear interpolation
        # precp = self.precp_series[idx] + (self.precp_series[idx + 1] - self.precp_series[idx]) \
        #     * (t - self.time_series[idx]) / (self.time_series[idx + 1] - self.time_series[idx]).unsqueeze(0)
        # temp = self.tmean_series[idx] + (self.tmean_series[idx + 1] - self.tmean_series[idx]) \
        #     * (t - self.time_series[idx]) / (self.time_series[idx + 1] - self.time_series[idx]).unsqueeze(0)
        # lday = self.lday_series[idx] + (self.lday_series[idx + 1] - self.lday_series[idx]) \
        #     * (t - self.time_series[idx]) / (self.time_series[idx + 1] - self.time_series[idx])

        # Interpolate the input variables
        t = t.detach().cpu().numpy()
        precp = self.precp_interp(t, extrapolate='periodic')
        temp = self.temp_interp(t, extrapolate='periodic')
        lday = self.lday_interp(t, extrapolate='periodic')

        # Convert to tensor
        precp = torch.tensor(precp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        temp = torch.tensor(temp, dtype=self.data_type_torch).unsqueeze(0).to(self.device)
        lday = torch.tensor(lday, dtype=self.data_type_torch)
        
        # Compute ET from the pretrainer.nnmodel
        inputs_nn = torch.stack([s0, s1, precp, temp], dim=-1)

        basin = self.basin
        if not isinstance(basin, list):
            basin = [basin]

        m100_outputs = self.pretrainer.nnmodel(inputs_nn, basin)[0] #.to(self.device)

        # Target variables:  Psnow, Prain, M, ET and, Q
        p_snow, p_rain, m, et, q = m100_outputs[0], m100_outputs[1], m100_outputs[2], \
                                   m100_outputs[3], m100_outputs[4]
        
        # Scale back to original values for the ODEs and Relu the Mechanism Quantities    
        if self.scale_target_vars:
            p_snow = torch.relu(torch.sinh(p_snow) * self.step_function(-temp[0]))
            p_rain = torch.relu(torch.sinh(p_rain))
            m = torch.relu(self.step_function(s0) * torch.sinh(m))
            et = self.step_function(s1) * torch.exp(et) * lday
            q = self.step_function(s1) * torch.exp(q) 

        # Eq 1 - Hoge_EtAl_HESS_2022
        ds0_dt = p_snow - m

        # Eq 2 - Hoge_EtAl_HESS_2022
        ds1_dt = p_rain + m - et - q

        return torch.stack([ds0_dt, ds1_dt], dim=-1)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        """
        Args:
            patience (int): How many epochs to wait after last time the monitored metric improved.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":
    pass