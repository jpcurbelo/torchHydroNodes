import os
import sys
from pathlib import Path
import argparse

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_dir)

from src.utils.utils_load_process import (
    load_forcing_target_data,
    Config,
)

from src.datasetzoo import get_dataset
from src.modelzoo_concept import get_concept_model

def main(config_file):
    
    # Create a Config object for the the run config
    cfg = Config(Path(config_file))
    
    # Load the forcing and target data 
    ds = get_dataset(cfg=cfg, is_train=True, scaler=dict()) 

    # Load the model
    model_concept = get_concept_model(cfg, ds.xr_train)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run file to test temporary code')
    parser.add_argument('--config-file', type=str, default='config_run.yml', help='Path to the config file')
    args = parser.parse_args()
    
    
    main(args.config_file)