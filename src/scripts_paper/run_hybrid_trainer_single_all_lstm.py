import os
import sys
from pathlib import Path
import yaml
import re
import torch
import concurrent.futures

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
script_path = Path(__file__).resolve().parent
project_path = str(script_path.parent.parent)
sys.path.append(project_path)

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
from utils import (
    get_basin_id,
    check_finished_basins,
    delete_unfinished_jobs,
)

# nnmodel_type = 'lstm'   # 'lstm' or 'mlp'

# pretrainer_runs_folder = None
# basin_file_all = '../../examples/569_basin_file.txt'

# config_file = Path(f'config_run_hybrid_{nnmodel_type}_single_cpu1.yml')
# # run_folder = f'569basins_single_{nnmodel_type}32x5_256b_euler1d_lr4_150ep_1000pre_lr3_carryoverYES'
# run_folder = f'569basins_single_{nnmodel_type}270d_128h_256b_euler1d_lr45_150ep_200pre_lr3'
# RUN_VERSION = 'lstm1'
# MAX_WORKERS = 16



NNMODEL_TYPE = 'lstm'   # 'lstm' or 'mlp'

CONFIG_FILE = Path(f'config_run_hybrid_{NNMODEL_TYPE}_single.yml')
# config_file = Path(f'config_run_hybrid_{NNMODEL_TYPE}_single_bosh3tol46_256b_50ep.yml')

# # #pretrainer_runs_folder = f'runs_pretrainer_single_{NNMODEL_TYPE}32x5'

PRETRAINER_RUN_FOLDER = None
BASIN_FILE_ALL = '../../examples/569_basin_file.txt'
# BASIN_FILE_ALL = '../../examples/4_basin_file.txt'

RUN_FOLDER = f'4basins_single_{NNMODEL_TYPE}32x5_256b_euler1d_lr4_50ep_1000pre_lr3'
# RUN_FOLDER = f'569basins_single_{NNMODEL_TYPE}32x5_256b_bosh3tol46_lr4_50ep_1000pre_lr3'

RUN_VERSION = 'lstm1'
# RUN_VERSION = 'bosh3tol46'

USE_PROCESS_POOL = 1
MAX_WORKERS = 4
# MAX_WORKERS = os.cpu_count()  # Adjust this based on your system and GPU availability

CHECK_IF_FINISHED = 1
DELETE_IF_UNFINISHED = 0

ONLY_CHECK_FINISHED = 0

def train_model_for_basin(config_file, nn_model_dir, pretrainer_runs_folder, project_path, run_folder, nnmodel_type, basin=None):
    '''
    Train the hybrid model for a single basin

    - Args:
        - nn_model_dir: str, neural network model directory
        - project_path: str, path to the project directory

    - Returns:
        - None
    '''

    # Load the MAIN configuration file
    if isinstance(config_file, Path):
        if config_file.exists():
            with open(config_file, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f'Configuration file {config_file} not found!')
        
    # Extract the basin name from the nn_model_dir
    if nn_model_dir is not None:
        basin = get_basin_id(nn_model_dir)

    # Create 1_basin_{basin}.txt file
    basin_file = f'1_basin_{basin}_{nnmodel_type}_{RUN_VERSION}.txt'
    with open(basin_file, 'w') as f:
        f.write(basin)

    # Update the configuration file with nn_model_dir and basin_file
    cfg['basin_file'] = basin_file
    if nn_model_dir is not None:
        cfg['nn_model_dir'] = pretrainer_runs_folder + '/' + nn_model_dir
        print(Path(project_path) / 'src' / 'scripts_paper' / pretrainer_runs_folder)
    else:
        # Remove the nn_model_dir from the configuration file
        cfg.pop('nn_model_dir', None)

    # Create temporary configuration file config_file_temp_basin.yml
    config_file_temp = str(config_file).split('.')[0] + f'_temp_{nnmodel_type}_{basin}_{RUN_VERSION}.yml'
    with open(config_file_temp, 'w') as f:
        yaml.dump(cfg, f)

    # Load the configuration file and dataset
    cfg_run, dataset = _load_cfg_and_ds(Path(config_file_temp), model='hybrid', 
                                        run_folder=run_folder,
                                        # nn_model_path=Path(project_path) / 'src' / 'scripts_paper' / pretrainer_runs_folder)
                                        nn_model_path=script_path)
    

    # Delete the basin_file and config_file_temp after training
    if os.path.isfile(basin_file):
        os.remove(basin_file)
    if os.path.isfile(config_file_temp):
        os.remove(config_file_temp)

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg_run, project_path)

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(cfg_run, dataset.ds_train, interpolators, time_idx0, 
                                      dataset.scaler)

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    # Load the neural network model state dictionary if cfg.nn_model_dir exists
    if cfg_run.nn_model_dir is not False:

        pattern = 'pretrainer_*basins.pth'

        model_path = Path(script_path) / cfg_run.nn_model_dir / 'model_weights'

        # Find the file(s) matching the pattern
        matching_files = list(model_path.glob(pattern))
        model_file = matching_files[0]
        # Load the neural network model state dictionary
        model_file = model_path / model_file
        # Load the state dictionary from the saved model
        state_dict = torch.load(model_file, map_location=torch.device(cfg_run.device))
        # Load the state dictionary into the model
        model_nn.load_state_dict(state_dict)

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # Pretrain the model if no pre-trained model is loaded
    if cfg_run.nn_model_dir is False:
        pretrain_ok = pretrainer.train(loss=cfg_run.loss_pretrain, lr=cfg_run.lr_pretrain, epochs=cfg_run.epochs_pretrain)

    if pretrain_ok:
        # Build the hybrid model
        model_hybrid = get_hybrid_model(cfg_run, pretrainer, dataset)

        # Build the trainer 
        trainer = get_trainer(model_hybrid)

        # Train the model
        trainer.train()
    else:
        print(f'Pretraining failed for basin {basin}')  

def main(config_file=CONFIG_FILE, pretrainer_runs_folder=PRETRAINER_RUN_FOLDER,
          basin_file_all=BASIN_FILE_ALL, run_folder=RUN_FOLDER, nnmodel_type=NNMODEL_TYPE):

    if pretrainer_runs_folder is not None:
        # Load available nn_model_dir in pretrainer_runs_folder
        nn_model_dirs = sorted([d for d in os.listdir(pretrainer_runs_folder) \
                        if os.path.isdir(os.path.join(pretrainer_runs_folder, d))])
        
        basins = [get_basin_id(nn_model_dir) for nn_model_dir in nn_model_dirs]

    else:
        print("pretrainer_runs_folder was not defined, then, pretrain on the fly")
        # Read the basin_file_all
        with open(basin_file_all, 'r') as f:
            basins = f.readlines()

        print(f"Total basins: {len(basins)}", basins[:5])
        nn_model_dirs = [None] * len(basins)
    
    if CHECK_IF_FINISHED and os.path.exists(script_path / run_folder):
            
            # Debugging step: Strip newline characters from basin IDs if present
            basins_str = sorted([str(int(basin.strip())) for basin in basins])
            
            # Find basins that are already finished and delete the unfinished jobs
            basin_finished, basin_unfinished = check_finished_basins(script_path / run_folder)

            print(f"Total basins: {len(basins_str)}")
            print(f"Finished basins: {len(basin_finished)}")
            print(f"Unfinished basins: {len(basin_unfinished)}", basin_unfinished)

            if DELETE_IF_UNFINISHED:
                delete_unfinished_jobs(script_path / run_folder, basin_unfinished)

            # print(f"Num Basins to continue: {len(basins_str) - len(basin_finished)}")
            # print('basins_str', basins_str[:5])
            # print('basin_finished', basin_finished[:5])

            # Remove the finished basins from the list
            # # basins_to_continue = [basin for basin in basins if basin.strip() not in basin_finished]
            basins_to_continue = [basin for basin in basins_str if basin not in basin_finished]

            print(f"Basins to continue: {len(basins_to_continue)}")

            # Update the list of basins
            basins = basins_to_continue

    else:
        basins = [str(int(basin)) for basin in basins]

    # Check if nn_model_dirs is not list of None
    if pretrainer_runs_folder is not None:
        # Filter the nn_model_dirs based on the basins
        nn_model_dirs = [dir for dir in nn_model_dirs[:] if str(int(get_basin_id(dir))) in basins]
    else:
        # 8 places leading zeros
        basins = sorted([str(int(basin)).zfill(8) for basin in basins])

    
    # basins = basins[:284]
    # basins = basins[284:]
    # # Reverse the list in place
    # basins.reverse()
    # # Now slice the reversed list
    # basins = basins[:128]

    # basins = basins[:68]
    # basins = basins[68:136]
    # basins = basins[136:204]
    # basins = basins[204:]

    print(f"Total basins to be trained: {len(basins)}")


    if ONLY_CHECK_FINISHED:
        return

    
    # Train the model for each basin
    if USE_PROCESS_POOL:

        max_workers = MAX_WORKERS
        # print(f'Number of workers: {max_workers}')

        if pretrainer_runs_folder is not None:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(train_model_for_basin, config_file, nn_model_dir, 
                                            pretrainer_runs_folder, project_path, 
                                            run_folder, nnmodel_type) 
                                            for nn_model_dir in nn_model_dirs]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Will raise exception if training failed
                    except Exception as e:
                        print(f'Error in training model: {e}')
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(train_model_for_basin, config_file, None, None, 
                                           project_path, run_folder, nnmodel_type, basin) 
                                           for basin in basins]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Will raise exception if training failed
                    except Exception as e:
                        print(f'Error in training model: {e}')

    else:
        if pretrainer_runs_folder is not None:
            for nn_model_dir in nn_model_dirs: 

                # print(nn_model_dir)
                # Extract the basin name from the nn_model_dir
                basin = str(int(get_basin_id(nn_model_dir)))
                # print('basin', basin, basin in basins)
                # aux = input('Continue?')

                if basin in basins:
                    print(nn_model_dir)
                    train_model_for_basin(config_file, nn_model_dir, pretrainer_runs_folder, project_path, 
                                          run_folder, nnmodel_type)
        else:
            for basin in basins:
                print(basin)
                train_model_for_basin(config_file, None, None, project_path, 
                                      run_folder, nnmodel_type, basin)


if __name__ == "__main__":

    main()