import os
import sys
from pathlib import Path
import argparse

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_dir)

from src.utils.utils_load_process import (
    load_run_config,
    load_forcing_target_data,
)

from src.datasetzoo import get_dataset

def main(config_file):
    
    # Load the run config file
    run_config = load_run_config(config_file)
    
    ds = get_dataset(cfg=run_config, period="train", is_train=True, scaler=dict())
    
    print('ds', ds)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run file to test temporary code')
    parser.add_argument('--config-file', type=str, default='config_run.yml', help='Path to the config file')
    args = parser.parse_args()
    
    
    main(args.config_file)