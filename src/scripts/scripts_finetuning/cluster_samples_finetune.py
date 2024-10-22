
import sys
from pathlib import Path
import yaml 
import concurrent.futures
import os
import logging

from utils import (
    validate_basin_file,
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

# nnmodel_type = 'mlp'   # 'lstm' or 'mlp'
nnmodel_type = 'lstm'   # 'lstm' or 'mlp'

# base_name = f'runs_finetune_{nnmodel_type}'
# base_name = f'test_runs_finetune_{nnmodel_type}'

SAMPLE_FRACTION = 0.1   # None
BASIN_FILE = '../../59_basin_file_sample_ok.txt'   # None 
# SAMPLE_FRACTION = 0.2   # None
# BASIN_FILE = '116_basin_file_sample.txt'  # None 
# CFG_FILE_BASE = Path('config_file_base_mlp.yml')
CFG_FILE_BASE = Path('config_file_base_lstm_4inp.yml')

# HP_FILE = 'hyperparameters_euler1d_lstm.yml'
# BASE_VERSION = 'euler1d_finetune_'

# HP_FILE = 'hyperparameters_euler1d_lstm_drop02nse.yml'
# BASE_VERSION = 'euler1d_finetune_drop02nse_'

# HP_FILE = 'hyperparameters_euler1d_lstm_drop02mse.yml'
# BASE_VERSION = 'euler1d_finetune_drop02mse_'

HP_FILE = 'hyperparameters_euler05d_lstm_drop02mse.yml'
BASE_VERSION = 'euler05d_finetune_drop02mse_'

# Remove trailing underscore if it exists
formatted_version = BASE_VERSION.rstrip('_')

# Create the base name using the formatted version
base_name = f'finetune_{nnmodel_type}_{formatted_version}_4inp'

FINETUNE_FOLDER = create_finetune_folder(base_name=base_name)

USE_PROCESS_POOL = 0
MAX_WORKERS = 1

# Setup dynamic logging for each run
def setup_logging(log_file):

    # Get the logger by name
    logger = logging.getLogger(str(log_file))
    logger.setLevel(logging.INFO)
    
    # Create a file handler for the log file
    file_handler = logging.FileHandler(str(log_file))
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Ensure that we don't add multiple handlers to the same logger
    if not logger.hasHandlers():
        logger.addHandler(file_handler)
    
    return logger

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

    if not pretrain_ok:
        print(f'Pretraining failed for basin {basin}')
        return False

    # Build the hybrid model
    try:
        model_hybrid = get_hybrid_model(cfg_run, pretrainer, dataset)
    except Exception as e:
        print(f'Error building hybrid model for basin {basin}: {e}')
        return False

    # Build the trainer 
    trainer = get_trainer(model_hybrid)

    # Train the model
    try:
        train_ok = trainer.train_finetune()
        if not train_ok:
            print(f'Training failed for basin {basin}')
            return False
    except Exception as e:
        print(f'Error training model for basin {basin}: {e}')
        return False

    return True  # Training succeeded

def main(basin_file=BASIN_FILE, sample_fraction=SAMPLE_FRACTION, config_file_base=CFG_FILE_BASE, 
         hyperparameter_file=HP_FILE, base_version=BASE_VERSION, finetune_folder=FINETUNE_FOLDER):
    
    # Check if basin_file is provided, exists and is not empty
    if not validate_basin_file(basin_file):
        print(f'Basin file {basin_file} not found or empty!')

        # Get the cluster files
        cluster_files = get_cluster_files()

        if len(cluster_files) == 0:
            raise FileNotFoundError('No cluster files found! Please, double-check the path.')

        # Random selection
        _, _, _, basin_file = random_basins_subset(cluster_files, sample_fraction)

    # print(f'Basin file: {basin_file}')

    # Read the basin_file_all
    with open(basin_file, 'r') as f:
         basins = [line.strip() for line in f.readlines()]

    # Load hyperparameters
    hyperparameters = load_hyperparameters(hyperparameter_file)

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
        cfg_file = run_folder / f'config_combo{i + 1}.yml'
        with open(cfg_file, 'w') as f:
            yaml.dump(cfg_run, f)

        run_version = f'{base_version}comb{i + 1}'

        # Set log file name dynamically per combination
        log_file = run_folder / f'log_combination_{i + 1}.log'
        logger = setup_logging(log_file)

        # Check if using parallel execution
        if USE_PROCESS_POOL:
            logger.info(f'Starting parallel training for combination {i + 1}: {combination}')
            max_workers = MAX_WORKERS
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_basin = {executor.submit(train_model_for_basin, run_folder, cfg_file, basin, run_version): 
                                   basin for basin in basins[:]}

                for future in concurrent.futures.as_completed(future_to_basin):
                    basin = future_to_basin[future]
                    try:
                        result = future.result()  # Get the result of the task
                        if result:
                            logger.info(f'Combination {i + 1}: training succeeded for basin {basin}')
                        else:
                            logger.error(f'Combination {i + 1}: training failed for basin {basin}')
                    except Exception as e:
                        logger.error(f'Combination {i + 1}: Error in training model for basin {basin}: {e}', exc_info=True)

        # Serial execution
        else:
            logger.info(f'Starting serial training for combination {i + 1}: {combination}')
            for basin in basins[:]:
                try:
                    logger.info(f'Starting training for basin {basin}')
                    result = train_model_for_basin(run_folder, cfg_file, basin, run_version)
                    if result:
                        logger.info(f'Combination {i + 1}: Training succeeded for basin {basin}')
                    else:
                        logger.error(f'Combination {i + 1}: Training failed for basin {basin}')
                except Exception as e:
                    logger.error(f'Combination {i + 1}: Error in training model for basin {basin}: {e}', exc_info=True)

        
if __name__ == "__main__":

    main(
        basin_file=BASIN_FILE, 
        sample_fraction=SAMPLE_FRACTION, 
        config_file_base=CFG_FILE_BASE, 
        hyperparameter_file=HP_FILE, 
        base_version=BASE_VERSION, 
        finetune_folder=FINETUNE_FOLDER
    )