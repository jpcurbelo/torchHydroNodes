import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import yaml
import xarray as xr
import numpy as np
from scipy.interpolate import Akima1DInterpolator

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(project_dir)

from src.utils.load_process_data import (
    Config,
    update_hybrid_cfg,
)
from src.datasetzoo import (
    get_dataset, 
    get_dataset_pretrainer
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
from src.utils.log_results import (
    save_and_plot_simulation,
    compute_and_save_metrics,
)

def _main():

    args = _get_args()
    
    if args["model"] == "conceptual":
        run_conceptual_model(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["model"] == "pretrainer" and args["action"] == "train":
        pretrain_nn_model(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["model"] == "hybrid" and args["action"] == "train":
        train_hybrid_model(config_file=Path(args["config_file"]), gpu=args["gpu"])
    elif args["action"] == "evaluate":
        evaluate_model(run_dir=Path(args["run_dir"]), period=args["period"], gpu=args["gpu"])
    elif args["action"] == "resume_training":
        # resume_training(run_dir=Path(args["run_dir"]), epoch=args["epoch"], gpu=args["gpu"])
        resume_training(run_dir=Path(args["run_dir"]))

def _get_args() -> dict:
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, choices=["conceptual", "pretrainer", "hybrid"],
                        help="Model to run")
    parser.add_argument("--action", type=str, choices=["train", "evaluate", "resume_training"],
                        default="train",
                        help="Action to perform")
    parser.add_argument('--config-file', type=str, default='config_run.yml', help='Path to the config file')
    parser.add_argument('--run-dir', type=str, 
                        help='Path to the directory where the run was saved - only for evaluate or resume_training mode')
    # parser.add_argument('--epoch', type=int, default=0, 
    #                     help='Epoch to resume training from or of which the model should be evaluated')
    parser.add_argument('--period', type=str, choices=['train', 'valid', 'test'],
                        default=['train', 'valid'], help='Period to evaluate the model')
    parser.add_argument('--gpu', type=int, default=0, 
                        help="GPU id to use. Overrides config argument 'device'. Use a value < 0 for CPU.")
    args = vars(parser.parse_args())
    
    if (args["model"] in ["conceptual", "pretrainer", "hybrid"] and args['action'] == "train") \
            and not args["config_file"]:
        raise ValueError("The config file is required to run the conceptual, pretrain, or train modes.")
    elif args["action"] in ["evaluate", "resume_training"] and not args["run_dir"]:
        raise ValueError("The run directory is required to evaluate or resume training the model.")
    
    return args

def _load_cfg_and_ds(config_file: Path, gpu: int = None, model: str = 'conceptual', 
                     run_folder='runs', nn_model_path=Path(project_dir) / 'data'):

    print('-- Loading the config file and the dataset')

    cfg = Config(config_file, run_folder=run_folder)

    if model in ['pretrainer', 'hybrid']:
        # Update the config file given the nn_model_dir
        cfg = update_hybrid_cfg(cfg, model, nn_model_path=nn_model_path)
    
    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        cfg.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        cfg.device = "cpu"

    # Load the forcing and target data 
    if model == 'conceptual':
        ds = get_dataset(cfg=cfg, is_train=True, scaler=dict()) 
    elif model == 'pretrainer':
        ds = get_dataset_pretrainer(cfg=cfg, scaler=dict()) 
    elif model == 'hybrid':
        ds = get_dataset_pretrainer(cfg=cfg, scaler=dict()) 
    else:
        raise ValueError("Invalid mode. Please specify 'conceptual' or 'pretrain'.")
    
    return cfg, ds

def _basin_interpolator_dict(ds, vars):
    '''
    Create interpolator functions for the input variables.

    - Args:
        ds: xarray.Dataset, dataset with the input variables.
        vars: list, list with the input variables to interpolate
    
    - Returns:
        interpolators: dict, dictionary with the interpolator functions for each variable.
    '''
    
    time_series = np.linspace(0, len(ds['date'].values) - 1, len(ds['date'].values))

    # Create a dictionary to store interpolator functions for each basin and variable
    interpolators = dict()
    
    # Loop over the basins and variables
    for var in vars:
                        
        # Get the variable values
        var_values = ds[var].values

        # Interpolate the variable values
        interpolators[var] = Akima1DInterpolator(time_series, var_values)
            
    return interpolators

def get_basin_interpolators(dataset, cfg, project_dir=project_dir):
    """
    Get the full dataset for interpolators and return the basin interpolators.

    Args:
        dataset: The dataset object containing period-specific datasets.
        cfg: Configuration object that includes the concept model name.
        project_dir: The root directory of the project.

    Returns:
        Dictionary of interpolators for the 
    """

    # Create a dictionary to store interpolator functions for each basin and variable
    interpolators = dict()

    for basin in dataset.basins:
        ## Get full dataset for interpolators
        # Get keys ds_* for the full dataset
        ds_periods = [key for key in dataset.__dict__.keys() if key.startswith('ds_') and 'static' not in key]

        # Extract the datasets using the keys
        datasets = [getattr(dataset, period).sel(basin=basin) for period in ds_periods]

        # Concatenate the datasets along the date dimension
        ds_full = xr.concat(datasets, dim='date')

        # Load interpolator_vars from utils/concept_model_vars.yml
        with open(Path(project_dir) / 'src' / 'utils' / 'concept_model_vars.yml', 'r') as f:
            var_alias = yaml.load(f, Loader=yaml.FullLoader)

        # Load the variables for the concept model
        interpolator_vars = var_alias[cfg.concept_model]['interpolator_vars']

        # Sort the concatenated dataset by the date dimension
        ds_full = ds_full.sortby('date')

        # Generate the interpolators
        interpolators[basin] = _basin_interpolator_dict(ds_full, interpolator_vars)

    return interpolators

def run_conceptual_model(config_file: Path, gpu: int = None):

    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model='conceptual')

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg, project_dir)
    
    print('-- Running the model and saving the results')
    for basin in tqdm(dataset.basins, disable=cfg .disable_pbar, file=sys.stdout):

        for period in dataset.start_and_end_dates.keys():
            
            # Extract the basin data
            if period == 'train':
                time_idx0 = 0
                model_concept = get_concept_model(cfg, dataset.ds_train, interpolators, time_idx0,
                                                dataset.scaler, odesmethod=cfg.odesmethod)
                basin_data = dataset.ds_train.sel(basin=basin)
            elif period == 'test':
                time_idx0 = len(dataset.ds_train['date'].values)
                model_concept = get_concept_model(cfg, dataset.ds_test, interpolators, time_idx0,
                                                  dataset.scaler, odesmethod=cfg.odesmethod)              
                basin_data = dataset.ds_test.sel(basin=basin)
            elif period == 'valid':
                time_idx0 = len(dataset.ds_train['date'].values)
                model_concept = get_concept_model(cfg, dataset.ds_valid, interpolators, time_idx0,
                                                  dataset.scaler, odesmethod=cfg.odesmethod)
                basin_data = dataset.ds_valid.sel(basin=basin)
            else:
                raise ValueError("Invalid period. Please specify 'train', 'test', or 'valid'.")

            # Update Initial states for the model if period is not 'train'
            if period != 'train':
                model_concept.shift_initial_states(dataset.start_and_end_dates, basin, period=period)
                
            # Run the model
            model_results = model_concept.run(basin=basin)

            # Save the results
            model_concept.save_results(basin_data, model_results, basin, period=period)
            
            # Plot the results 
            save_and_plot_simulation(ds=basin_data,
                                q_bucket=model_results[-1],
                                basin=basin,
                                period=period,
                                model_name=cfg.concept_model,
                                plots_dir=cfg.plots_dir,
                                plot_prcp=False
                            )
            
    run_dir = cfg.run_dir
    # run_dir = Path('../examples/runs/concept_run_240506_183950')
    ## After the model has been run for all basins and periods - Evaluate the model
    # Compute the metrics
    compute_and_save_metrics( metrics=cfg.metrics, run_dir=run_dir)

def pretrain_nn_model(config_file: Path, gpu: int = None):
    
    # Load the configuration file and dataset
    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model='pretrainer')

    # print('dataset:', dataset.__dict__.keys())
    # print('dataset.scaler', dataset.scaler)
    # print('dataset.cfg._cfg.keys()', dataset.cfg._cfg.keys())
    # # if 'static_attributes' in dataset.cfg._cfg:
    # print('self.ds_static', dataset.ds_static)

    # print('************************')

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg, project_dir)

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(cfg, dataset.ds_train, interpolators, time_idx0, 
                                      dataset.scaler)
    
    # print('model_concept:', model_concept.__dict__.keys())

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # Train the model
    pretrainer.train()

def train_hybrid_model(config_file: Path, gpu: int = None):
        
    # Load the configuration file and dataset
    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model='hybrid')

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg, project_dir)

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(cfg, dataset.ds_train, interpolators, time_idx0,
                                        dataset.scaler)

    print(f'-- Conceptual model: {model_concept.__class__.__name__}')

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    print(f'-- Neural network model: {model_nn.__class__.__name__}')
    # print(model_nn)

    # Load the neural network model state dictionary if cfg.nn_model_dir exists
    if cfg.nn_model_dir is not False:

        print(f'-- Loading the neural network model state dictionary from {cfg.nn_model_dir}')

        pattern = 'pretrainer_*basins.pth'
        model_path = Path(project_dir) / 'data' / cfg.nn_model_dir / 'model_weights'
        # Find the file(s) matching the pattern
        matching_files = list(model_path.glob(pattern))
        model_file = matching_files[0]
        # model_file = '1013500_modelM100_leakyrelu.pth'
        print(f'-- Loading the model weights from {model_file}')
        # Load the neural network model state dictionary
        model_file = model_path / model_file
        # Load the state dictionary from the saved model
        state_dict = torch.load(model_file, map_location=torch.device(cfg.device))
        # print('state_dict:', state_dict)
        # Load the state dictionary into the model
        model_nn.load_state_dict(state_dict)

    # aux = input("Press Enter to continue...")

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # # # Pretrain the model if no pre-trained model is loaded
    # # if cfg.nn_model_dir is False:
    # #     pretrainer.train()

    # Build the hybrid model
    model_hybrid = get_hybrid_model(cfg, pretrainer, dataset)

    # Build the trainer 
    trainer = get_trainer(model_hybrid)
    # Train the model
    trainer.train()

def evaluate_model(run_dir: Path, period: str, gpu: int=None, 
                   config_file=None, model='pretrainer', epoch: int=-1):

    if config_file is None:
        config_file = run_dir / 'config.yml'

    # Load the configuration file and dataset
    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model=model)

    # Get the basin interpolators
    interpolators = get_basin_interpolators(dataset, cfg, project_dir)

    # Conceptual model
    time_idx0 = 0
    model_concept = get_concept_model(cfg, dataset.ds_train, interpolators, time_idx0,
                                        dataset.scaler)

    # Neural network model
    model_nn = get_nn_model(model_concept, dataset.ds_static)

    model_path = run_dir / 'model_weights'
    basins = dataset.basins
    # Load the neural network model state dictionary
    model_file = model_path / f'pretrainer_{cfg.nn_model}_{len(basins)}basins.pth'
    # Load the state dictionary from the saved model
    state_dict = torch.load(model_file)
    # Load the state dictionary into the model
    model_nn.load_state_dict(state_dict)

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # Train the model
    pretrainer.evaluate()

def resume_training(run_dir: Path, epoch: int = None, gpu: int = None):

    # Load the config file
    config_file = run_dir / 'config.yml'
    cfg = Config(config_file)
    if 'hybrid_model' in cfg._cfg:
        model_type = 'hybrid'
    else:
        model_type = 'pretrainer'
    
    # # Load the configuration file and dataset
    # cfg, dataset = _load_cfg_and_ds(config_file, gpu, model=model_type)

    # # Load config_resume file
    # config_resume_file = run_dir / 'config_resume.yml'
    # if config_resume_file.exists():
    #     with open(config_resume_file, 'r') as ymlfile:
    #         cfg_resume = yaml.load(ymlfile, Loader=yaml.FullLoader)
    # else:
    #     raise FileNotFoundError(f"File not found: {config_resume_file} (mandatory for resuming training)")
                                
    # # Update the config file
    # cfg._cfg.update(cfg_resume)

    # # Define and create resume_folder
    # resume_folder = run_dir / 'resume'
    # resume_folder.mkdir(parents=True, exist_ok=True)

    # # Update model_plots, model_results, and model_weights and create folders
    # cfg.plots_dir = resume_folder / 'model_plots'
    # cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    # cfg.results_dir = resume_folder / 'model_results'
    # cfg.results_dir.mkdir(parents=True, exist_ok=True)
    # cfg.weights_dir = resume_folder / 'model_weights'
    # cfg.weights_dir.mkdir(parents=True, exist_ok=True)

    # # print(f'-- Updated config file: {cfg._cfg}')

    # # Conceptual model
    # model_concept = get_concept_model(cfg, dataset.ds_train, dataset.scaler)

    # print(f'-- Conceptual model: {model_concept.__class__.__name__}')

    # # Neural network model
    # model_nn = get_nn_model(model_concept)

    # # # Print state dictionary
    # # print(model_nn.state_dict())

    # print(f'-- Neural network model: {model_nn.__class__.__name__}')

    # Load the neural network model state dictionary if run_dir/model_weights/*.pth exists
    model_path = run_dir / 'model_weights'
    # Check if a file *.pth exists in the model_weights folder
    matching_files = list(model_path.glob('*.pth'))
    if len(matching_files) > 0:
        model_file = matching_files[0]
        # Load the state dictionary from the saved model
        state_dict = torch.load(model_file)
        print('state_dict:', state_dict)
        # # Load the state dictionary into the model
        # model_nn.load_state_dict(state_dict)
        # print(f'-- Loaded the model weights from {model_file}')
    else:
        print(f'-- No model weights found in {model_path}')

    # # Pretrainer
    # pretrainer = get_nn_pretrainer(model_nn, dataset)

    # # Build the hybrid model
    # model_hybrid = get_hybrid_model(cfg, pretrainer, dataset)

    # # Build the trainer 
    # trainer = get_trainer(model_hybrid)
    # # Train the model
    # trainer.train(is_resume=True)


    

# Example usage:
# python thn_run.py conceptual --config-file ../examples/config_run_m0.yml
# python thn_run.py pretrainer --action train --config-file ../examples/config_run_nn_test.yml
# python thn_run.py pretrainer --action train --config-file ../examples/config_run_nn_mlp.yml
# python thn_run.py pretrainer --action train --config-file ../examples/config_run_nn_lstm.yml
# python thn_run.py pretrainer --action train --config-file ../examples/config_run_nn_cluster_lstm.yml
# python thn_run.py pretrainer --action train --config-file ../examples/config_run_nn_cluster_mlp.yml

# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid.yml
# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid1basin.yml
# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid1basin_test.yml
# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid4basins.yml
# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid4basins_mlp.yml
# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid4basins_lstm.yml

# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid_cluster_lstm.yml
# python thn_run.py hybrid --action train --config-file ../examples/config_run_hybrid_cluster_mlp.yml

# python thn_run.py pretrainer --action evaluate --run-dir ../examples/runs/pretrainer_run_240530_105452
# python thn_run.py hybrid --action resume_training --run-dir ../examples/runs/1basin_hybrid_lstm_06431500_240704_125621

if __name__ == "__main__":
    _main()