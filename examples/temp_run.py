import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_dir)

from src.utils.load_process import (
    Config,
)

from src.datasetzoo import get_dataset
from src.modelzoo_concept import get_concept_model
from src.modelzoo_nn import get_nn_model

def main(config_file):
    
    # Create a Config object for the the run config
    cfg = Config(Path(config_file))
    
    # Load the forcing and target data 
    dataset = get_dataset(cfg=cfg, is_train=True, scaler=dict()) 

    model_concept = get_concept_model(cfg, dataset.ds_train, scaler=dataset.scaler)

    # print('nn_dynamic_inputs',cfg.nn_dynamic_inputs)
    # print('nn_outputs', model_concept.nn_outputs)
    # print('model_outputs', model_concept.model_outputs)

    model_nn = get_nn_model(model_concept)

    print(model_nn)
    

if __name__ == '__main__':
    
    # python temp_run.py --config-file config_run.yml
    parser = argparse.ArgumentParser(description='Run file to test temporary code')
    parser.add_argument('--config-file', type=str, default='config_run.yml', help='Path to the config file')
    args = parser.parse_args()
    
    
    main(args.config_file)