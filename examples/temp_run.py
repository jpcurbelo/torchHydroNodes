import os
import sys
from pathlib import Path
import argparse
from tqdm import tqdm
import torch

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_dir)

from src.utils.load_process_data import (
    Config,
)

from src.datasetzoo import (
    get_dataset,
    get_dataset_pretrainer,
)

from src.modelzoo_concept import get_concept_model
from src.modelzoo_nn import (
    get_nn_model,
    get_nn_pretrainer,
)

def main(config_file):
    
    # Create a Config object for the the run config
    cfg = Config(Path(config_file))
    
    # Load the forcing and target data 
    dataset = get_dataset_pretrainer(cfg=cfg, scaler=dict()) 

    # print(dataset.__dict__)
    # print(dataset.__dict__['alias_map_clean'])

    ### Conceptual model
    model_concept = get_concept_model(cfg, dataset.ds_train, dataset.scaler)

    # print('\n')
    # print(model_concept.__dict__)
    # print('\n')

    # print('nn_dynamic_inputs',cfg.nn_dynamic_inputs)
    # print('nn_outputs', model_concept.nn_outputs)
    # print('model_outputs', model_concept.model_outputs)

    ### Neural network model
    model_nn = get_nn_model(model_concept)

    # # Prepare data to test the model
    # basin = '01022500'
    # inputs = torch.cat([torch.tensor(dataset.ds_train[var.lower()].sel(basin=basin).values).unsqueeze(0) \
    #                     for var in cfg.nn_dynamic_inputs], dim=0).t().to(model_nn.device)
    # output = model_nn(inputs, basin)

    # print(output)

    ### Pretrainer
    nn_dynamic_inputs = cfg.nn_dynamic_inputs
    nn_outputs = model_concept.nn_outputs
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    ## Train the model
    pretrainer.train()




if __name__ == '__main__':
    
    # python temp_run.py --config-file config_run_nn_pre.yml
    parser = argparse.ArgumentParser(description='Run file to test temporary code')
    parser.add_argument('--config-file', type=str, default='config_run.yml', help='Path to the config file')
    args = parser.parse_args()
    
    main(args.config_file)