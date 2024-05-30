import sys
import argparse
from pathlib import Path
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
    get_dataset_pretrainer
)
from src.modelzoo_concept import get_concept_model
from src.modelzoo_nn import (
    get_nn_model,
    get_nn_pretrainer,
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
        resume_training(run_dir=Path(args["run_dir"]), epoch=args["epoch"], gpu=args["gpu"])

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

def _load_cfg_and_ds(config_file: Path, gpu: int = None, model: str = 'conceptual'):

    print('-- Loading the config file and the dataset')

    cfg = Config(config_file)
    
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
    else:
        raise ValueError("Invalid mode. Please specify 'conceptual' or 'pretrain'.")
    
    return cfg, ds

def run_conceptual_model(config_file: Path, gpu: int = None):

    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model='conceptual')
    
    print('-- Running the model and saving the results')
    for basin in tqdm(dataset.basins, disable=cfg .disable_pbar, file=sys.stdout):

        for period in dataset.start_and_end_dates.keys():
            
            # Extract the basin data
            if period == 'train':
                model_concept = get_concept_model(cfg, dataset.ds_train, dataset.scaler)
                basin_data = dataset.ds_train.sel(basin=basin)
            elif period == 'test':
                model_concept = get_concept_model(cfg, dataset.ds_test, dataset.scaler)              
                basin_data = dataset.ds_test.sel(basin=basin)
            elif period == 'valid':
                model_concept = get_concept_model(cfg, dataset.ds_valid, dataset.scaler)
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
    
    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model='pretrainer')

    # Conceptual model
    model_concept = get_concept_model(cfg, dataset.ds_train, dataset.scaler)

    # Neural network model
    model_nn = get_nn_model(model_concept)

    # Pretrainer
    pretrainer = get_nn_pretrainer(model_nn, dataset)

    # Train the model
    pretrainer.train()

def train_hybrid_model(config_file: Path, gpu: int = None):
        
    pass

def evaluate_model(run_dir: Path, period: str, gpu: int=None, 
                   config_file=None, model='pretrainer', epoch: int=-1):

    if config_file is None:
        config_file = run_dir / 'config.yml'

    cfg, dataset = _load_cfg_and_ds(config_file, gpu, model=model)

    # Conceptual model
    model_concept = get_concept_model(cfg, dataset.ds_train, dataset.scaler)

    # Neural network model
    model_nn = get_nn_model(model_concept)

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

def resume_training(run_dir: Path, epoch: int, gpu: int = None):

    pass

# Example usage:
# python thn_run.py conceptual --config-file ../examples/config_run_m0.yml
# python thn_run.py pretrainer --action train --config-file ../examples/config_run_nn_pre.yml
# python thn_run.py pretrainer --action evaluate --run-dir ../examples/runs/pretrainer_run_240530_105452
if __name__ == "__main__":
    _main()