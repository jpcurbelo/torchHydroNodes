
import sys
from pathlib import Path
import yaml 
import concurrent.futures
import os
import logging

from utils import (
    random_basins_subset,
    load_hyperparameters,
    hyperparameter_combinations,
    create_finetune_folder,
)

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
script_path = Path(__file__).resolve().parent
project_path = str(script_path.parent.parent.parent)
sys.path.append(project_path)

from src.utils.plots import (
    get_cluster_files,
)

from src.thn_run import (
    _load_cfg_and_ds,
    get_basin_interpolators,
)
from src.modelzoo_concept import get_concept_model
from src.modelzoo_nn import (
    get_nn_model,
    get_nn_pretrainer,
)
from src.modelzoo_hybrid import (
    get_hybrid_model,
    get_trainer,
)

nnmodel_type = 'mlp'   # 'lstm' or 'mlp'

SAMPLE_FRACTION = 0.01
config_file_base = Path(f'config_file_base_{nnmodel_type}_testing.yml')
hyperparameter_file = f'hyperparameters_{nnmodel_type}_testing.yml'
BASE_VERSION = ''
base_name = f'test_runs_finetune_{nnmodel_type}_{BASE_VERSION.split("_")[0]}'

# SAMPLE_FRACTION = 0.1
# config_file_base = Path(f'config_file_base_{nnmodel_type}.yml')
# hyperparameter_file = f'hyperparameters_{nnmodel_type}.yml'
# BASE_VERSION = 'v1'
# base_name = f'runs_finetune_{nnmodel_type}_{BASE_VERSION.split("_")[0]}'


finetune_folder = create_finetune_folder(base_name=base_name)

USE_PROCESS_POOL = 1
MAX_WORKERS = 4

def train_model_for_basin(run_folder, cfg_file, basin, run_version):
    '''
    Train the hybrid model for a single basin
    '''

    # Load the MAIN configuration file
    if isinstance(cfg_file, Path):
        if cfg_file.exists():
            with open(cfg_file, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f'Configuration file {cfg_file} not found!')

    # Create 1_basin_{basin}_{run_version}.txt file
    basin_file = Path(f'1_basin_{basin}_{run_version}.txt')
    with open(basin_file, 'w') as f:
        f.write(basin)

    # Update the configuration file with basin_file
    cfg['basin_file'] = str(basin_file).split('/')[-1]

    # Create the temporary configuration file within the correct folder
    config_file_temp = Path(f'config_temp_{basin}_{run_version}.yml')

    # Write the config file
    with open(config_file_temp, 'w') as f:
        yaml.dump(cfg, f)

    # Load the configuration file and dataset
    cfg_run, dataset = _load_cfg_and_ds(
        config_file_temp, 
        model='hybrid', 
        run_folder=str(run_folder), 
        nn_model_path=script_path
    )

    # Delete basin_file and config_file_temp
    if os.path.isfile(basin_file):
        basin_file.unlink()
    if os.path.isfile(config_file_temp):
        config_file_temp.unlink()

    # Get the basin interpolators
    interpolators = get_basin_interpolators(
        dataset, 
        cfg_run, 
        project_path
    )

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(
        cfg_run, 
        dataset.ds_train, 
        interpolators, 
        time_idx0,
        dataset.scaler
    )

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # Pretrain the model
    pretrain_ok = pretrainer.train(
        loss=cfg_run.loss_pretrain, 
        lr=cfg_run.lr_pretrain, 
        epochs=cfg_run.epochs_pretrain,
        any_log=False)

    if pretrain_ok:
        # Build the hybrid model
        model_hybrid = get_hybrid_model(cfg_run, pretrainer, dataset)

        # Build the trainer 
        trainer = get_trainer(model_hybrid)

        # Train the model
        trainer.train_finetune()
    else:
        print(f'Pretraining failed for basin {basin}') 




def main():

    # Get the cluster files
    cluster_files = get_cluster_files()

    if len(cluster_files) == 0:
        raise FileNotFoundError('No cluster files found! Please, double-check the path.')

    # Random selection
    selected_basins_dict, _, sample_file, basin_file = random_basins_subset(cluster_files, SAMPLE_FRACTION)

    # Read the basin_file_all
    with open(basin_file, 'r') as f:
         basins = [line.strip() for line in f.readlines()]

    # Load hyperparameters
    hyperparameters = load_hyperparameters(hyperparameter_file)
    # print(hyperparameters)

    # Generate hyperparameter combinations`
    params_combinations = hyperparameter_combinations(hyperparameters)

    # Load base configuration file
    if isinstance(config_file_base, Path):
        if config_file_base.exists():
            with open(config_file_base, 'r') as f:
                cfg_base = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f'Configuration file {config_file_base} not found!')      

    # Iterate over the hyperparameter combinations
    for i, combination in enumerate(params_combinations):
        # print(f"Combination {i + 1}: {combination}")

        run_folder = finetune_folder / f"run_combination{i + 1}"
        run_folder.mkdir(parents=True, exist_ok=True)

        # Save the combination to a YAML file
        combination_file = run_folder / f'hyperparameters_comb{i + 1}.yml'
        with open(combination_file, 'w') as f:
            yaml.dump(combination, f)

        # Update the base configuration with the combination
        cfg_run = cfg_base.copy()
        cfg_run.update(combination)
        # # Set log_every_n_epochs to be half of the number of epochs
        # cfg_run['log_every_n_epochs'] = cfg_run['epochs'] // 2

        # Save the combination to a YAML file
        cfg_file = run_folder / f'config_comb{i + 1}.yml'
        with open(cfg_file, 'w') as f:
            yaml.dump(cfg_run, f)

        run_version = f'{BASE_VERSION}comb{i + 1}'

        # Train the model for each basin
        if USE_PROCESS_POOL:

            # Setup logging
            logging.basicConfig(level=logging.INFO)

            max_workers = MAX_WORKERS
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_basin = {executor.submit(train_model_for_basin, run_folder, cfg_file, basin, run_version): basin for basin in basins}

                for future in concurrent.futures.as_completed(future_to_basin):
                    basin = future_to_basin[future]
                    try:
                        future.result()  # Raises exception if the task failed
                    except Exception as e:
                        logging.error(f'Error in training model for basin {basin}: {e}', exc_info=True)
        else:
            print(f"Training model for combination {i + 1}")
            for basin in basins[:]:
                print(f"Training model for basin {basin}")
                train_model_for_basin(run_folder, cfg_file, basin, run_version)

        






if __name__ == "__main__":

    main()