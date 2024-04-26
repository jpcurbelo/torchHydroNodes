import yaml
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import List

## Functions
def load_run_config(config_file):
    '''
    Load the configuration data from a YAML file, save the configuration data to a JSON file,
    and return the configuration data.
    
    - Args:
        config_file: str, path to the YAML file with the configuration data.
        
    - Returns:
        config_data: dict, configuration data.
    '''
    
    ## Config data
    with open(config_file, 'r') as ymlfile:
        config_data = yaml.load(ymlfile, Loader=yaml.FullLoader)  
         
    if "experiment_name" in config_data and config_data["experiment_name"]:
        experiment_name = config_data['experiment_name']
        if not os.path.exists(experiment_name):
            os.mkdir(experiment_name)
                            
        # Create a folder to save the trained models
        now = datetime.now()
        dt_string = now.strftime("%y%m%d %H%M%S")
        run_dir = os.path.join(experiment_name  , f'run_{dt_string.split()[0]}_{dt_string.split()[1]}')
        try:
            os.mkdir(run_dir)
        except OSError as error:
            print(f"Folder '{run_dir}' already existed.")

     
    # Save the configuration data to a JSON file   
    config_data_file = run_dir + '/config_data.json'
    with open(config_data_file, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    # Transform dates to datetime objects
    config_data['train_start_date'] = pd.to_datetime(config_data['train_start_date'], format='%d/%m/%Y')
    config_data['train_end_date'] = pd.to_datetime(config_data['train_end_date'], format='%d/%m/%Y')
    config_data['valid_start_date'] = pd.to_datetime(config_data['valid_start_date'], format='%d/%m/%Y')
    config_data['valid_end_date'] = pd.to_datetime(config_data['valid_end_date'], format='%d/%m/%Y')
    config_data['test_start_date'] = pd.to_datetime(config_data['test_start_date'], format='%d/%m/%Y')
    config_data['test_end_date'] = pd.to_datetime(config_data['test_end_date'], format='%d/%m/%Y')
    
    # Transform data_dir to a Path object
    config_data['data_dir'] = Path(config_data['data_dir'])
    
    return config_data        
          
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
        
    return basins      
          
          
                    
def load_forcing_target_data(run_config):
    
    pass



if __name__ == "__main__":
    pass