import os
import sys
from pathlib import Path
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

from utils import (
    check_finished_basins,
    delete_unfinished_jobs,
)

# basin_file_all = '../../examples/4_basin_file.txt'
basin_file_all = '../../examples/569_basin_file.txt'

nnmodel_type = 'lstm'   # 'lstm' or 'mlp'

# config_file = Path(f'config_run_nn_{nnmodel_type}_single.yml')
# run_folder = f'runs_pretrainer_single_{nnmodel_type}'

config_file = Path(f'config_run_nn_{nnmodel_type}_single270.yml')
run_folder = f'runs_pretrainer_single_{nnmodel_type}270'

MAX_WORKERS = 1

CHECK_IF_FINISHED = 1
DELETE_IF_UNFINISHED = 0

def train_model_for_basin(basin, config_file, project_path):
    '''
    Train the model for a single basin
    
    - Args:
        - basin: str, basin name
        - config_file: Path, path to the configuration file
        - project_path: str, path to the project directory
    
    - Returns:
        - None
    '''

    # Create 1_basin.txt file
    basin_file = f'1_basin_{nnmodel_type}.txt'
    with open(basin_file, 'w') as f:
        f.write(basin)

    # Load the configuration file and dataset
    cfg, dataset = _load_cfg_and_ds(config_file, model='pretrainer', run_folder=run_folder)

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg, project_path)

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(cfg, dataset.ds_train, interpolators, time_idx0, dataset.scaler)

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # Train the model
    pretrainer.train()

####################################################################################################
def main():
    
    if os.path.exists(basin_file_all):
        with open(basin_file_all, 'r') as f:
            basins = f.readlines()
            
        # # max_workers = os.cpu_count()  # Adjust this based on your system and GPU availability
        # max_workers = MAX_WORKERS

        # print(f"Training models for {len(basins)} basins using {max_workers} workers.")

        if CHECK_IF_FINISHED and os.path.exists(script_path / run_folder):

            # Find basins that are already finished and delete the unfinished jobs
            basin_finished, basin_unfinished = check_finished_basins(script_path / run_folder)

            print(f"Total basins: {len(basins)}")
            print(f"Finished basins: {len(basin_finished)}")
            print(f"Unfinished basins: {len(basin_unfinished)}", basin_unfinished)

            if DELETE_IF_UNFINISHED:
                delete_unfinished_jobs(script_path / run_folder, basin_unfinished)

            # Remove the finished basins from the list
            basins_to_continue = [basin for basin in basins if basin.strip() not in basin_finished]

            print(f"Basins to continue: {len(basins_to_continue)}")

            # Update the list of basins
            basins = basins_to_continue


        # # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # #     futures = [executor.submit(train_model_for_basin, basin, config_file, project_path) for basin in basins]
        # #     for future in concurrent.futures.as_completed(futures):
        # #         try:
        # #             future.result()  # Will raise exception if training failed
        # #         except Exception as e:
        # #             print(f"Training failed for a basin: {e}")

        # for basin in basins[:]:
        #     train_model_for_basin(basin, config_file, project_path)

    else:
        print(f"File {basin_file_all} not found.")
        sys.exit(1)


if __name__ == "__main__":

    main()