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

nnmodel_type = 'lstm'   # 'lstm' or 'mlp'

config_file = Path(f'config_run_hybrid_{nnmodel_type}_single.yml')

# # pretrainer_runs_folder = f'runs_pretrainer_single_{nnmodel_type}32x5'
# pretrainer_runs_folder = f'runs_pretrainer_single_{nnmodel_type}32x5_7304b_lr2_200ep'
# # run_folder = f'runs_hybrid_single_{nnmodel_type}32x5_7304b_bosh3_lr34_200ep'
# # run_folder = f'runs_hybrid_single_{nnmodel_type}32x5_7304b_bosh3_lr4_100ep'
# run_folder = f'runs_hybrid_single_{nnmodel_type}32x5_7304b_euler_lr4_150ep'
# # run_folder = f'runs_hybrid_single_{nnmodel_type}32x5_7304b_euler'

# # # pretrainer_runs_folder = f'runs_pretrainer_single_{nnmodel_type}365_128'
# # # run_folder = f'runs_hybrid_single_{nnmodel_type}365d_128h_256b'

# pretrainer_runs_folder = f'runs_pretrainer_single_{nnmodel_type}270d_128h'
pretrainer_runs_folder = f'runs_pretrainer_single_{nnmodel_type}270d_128h_7036b_lr3_200ep'
run_folder = f'runs_hybrid_single_{nnmodel_type}270d_128h_7036b_bosh3_lr34_100ep'
# run_folder = f'runs_hybrid_single_{nnmodel_type}270d_128h_256b_new_temp'

USE_PROCESS_POOL = 0
MAX_WORKERS = 64
# MAX_WORKERS = os.cpu_count()  # Adjust this based on your system and GPU availability

CHECK_IF_FINISHED = 1
DELETE_IF_UNFINISHED = 1

def train_model_for_basin(nn_model_dir, project_path):
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
    basin = get_basin_id(nn_model_dir)

    # Create 1_basin_{basin}.txt file
    basin_file = f'1_basin_{basin}_{nnmodel_type}.txt'
    with open(basin_file, 'w') as f:
        f.write(basin)

    # Update the configuration file with nn_model_dir and basin_file
    cfg['nn_model_dir'] = pretrainer_runs_folder + '/' +nn_model_dir
    cfg['basin_file'] = basin_file

    # Create temporary configuration file config_file_temp_basin.yml
    config_file_temp = str(config_file).split('.')[0] + f'_temp_{basin}.yml'
    with open(config_file_temp, 'w') as f:
        yaml.dump(cfg, f)

    # Load the configuration file and dataset
    print(Path(project_path) / 'src' / 'scripts_paper' / pretrainer_runs_folder)
    cfg_run, dataset = _load_cfg_and_ds(Path(config_file_temp), model='hybrid', 
                                        run_folder=run_folder,
                                        # nn_model_path=Path(project_path) / 'src' / 'scripts_paper' / pretrainer_runs_folder)
                                        nn_model_path=script_path)

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg_run, project_path)

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(cfg_run, dataset.ds_train, interpolators, time_idx0, 
                                      dataset.scaler)

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    # Load the neural network model state dictionary if cfg.nn_model_dir exists
    if cfg_run.nn_model_dir is not None:

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

    # Build the hybrid model
    model_hybrid = get_hybrid_model(cfg_run, pretrainer, dataset)

    # Build the trainer 
    trainer = get_trainer(model_hybrid)

    # Train the model
    trainer.train()

    # Delete the basin_file and config_file_temp after training
    if os.path.isfile(basin_file):
        os.remove(basin_file)
    if os.path.isfile(config_file_temp):
        os.remove(config_file_temp)

def main():

    # Load available nn_model_dir in pretrainer_runs_folder
    nn_model_dirs = sorted([d for d in os.listdir(pretrainer_runs_folder) \
                    if os.path.isdir(os.path.join(pretrainer_runs_folder, d))])

    basins = [get_basin_id(nn_model_dir) for nn_model_dir in nn_model_dirs]
    
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

            print(f"Num Basins to continue: {len(basins_str) - len(basin_finished)}")
            print('basins_str', basins_str[:5])
            print('basin_finished', basin_finished[:5])

            # Remove the finished basins from the list
            # # basins_to_continue = [basin for basin in basins if basin.strip() not in basin_finished]
            basins_to_continue = [basin for basin in basins_str if basin not in basin_finished]

            print(f"Basins to continue: {len(basins_to_continue)}")

            # Update the list of basins
            basins = basins_to_continue

    else:
        basins = [str(int(basin)) for basin in basins]

    # Filter the nn_model_dirs based on the basins
    nn_model_dirs = [dir for dir in nn_model_dirs[:] if str(int(get_basin_id(dir))) in basins]

    print(f"Total basins to be trained: {len(nn_model_dirs)}")
    
    # Train the model for each basin
    if USE_PROCESS_POOL:

        max_workers = MAX_WORKERS
        # print(f'Number of workers: {max_workers}')

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(train_model_for_basin, nn_model_dir, project_path) for nn_model_dir in nn_model_dirs]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()  # Will raise exception if training failed
                except Exception as e:
                    print(f'Error in training model: {e}')

    else:
        for nn_model_dir in nn_model_dirs[:20]: 

            # print(nn_model_dir)
            # Extract the basin name from the nn_model_dir
            basin = str(int(get_basin_id(nn_model_dir)))
            # print('basin', basin, basin in basins)
            # aux = input('Continue?')

            if basin in basins:
                print(nn_model_dir)
                train_model_for_basin(nn_model_dir, project_path)


if __name__ == "__main__":

    main()