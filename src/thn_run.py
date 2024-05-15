import sys
import argparse
from pathlib import Path
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
from src.utils.log_results import (
    save_and_plot_simulation,
    compute_and_save_metrics,
)

def _main():
    args = _get_args()
    
    if args["mode"] == "train":
        start_run_m0(config_file=Path(args["config_file"]), gpu=args["gpu"])

def _get_args() -> dict:
    
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, default="train", 
                        choices=["train", "evaluate", "resume_training"],
                        help="Mode to run the model")
    parser.add_argument('--config-file', type=str, default='config_run.yml', 
                        help='Path to the config file')
    parser.add_argument('--run-dir', type=str, 
                        help='Path to the directory where the run was saved - only for evaluate or resume_training mode')
    parser.add_argument('--epoch', type=int, default=0, 
                        help='Epoch to resume training from or of which the model should be evaluated')
    parser.add_argument('--period', type=str, default='test', 
                        choices=['train', 'valid', 'test'],
                        help='Period to evaluate the model')
    parser.add_argument('--gpu', type=int, default=0, 
                        help="GPU id to use. Overrides config argument 'device'. Use a value < 0 for CPU.")
    args = vars(parser.parse_args())
    
    if args["mode"] == "train" and (args["config_file"] is None):
        raise ValueError("The config file is required to train the model.")
    elif args["mode"] in ["evaluate", "resume_training"] and (args["run_dir"] is None):
        raise ValueError("The run directory is required to evaluate or resume training the model.")
    
    return args

def _load_cfg_and_ds(config_file: Path, gpu: int = None):

    cfg = Config(config_file)
    
    # check if a GPU has been specified as command line argument. If yes, overwrite config
    if gpu is not None and gpu >= 0:
        print('Using GPU:', gpu)
        print('Device in config:', cfg.device)
        cfg.device = f"cuda:{gpu}"
    if gpu is not None and gpu < 0:
        cfg.device = "cpu"

    # Load the forcing and target data 
    ds = get_dataset(cfg=cfg, is_train=True, scaler=dict()) 
    
    return cfg, ds


def start_run_m0(config_file: Path, gpu: int = None):

    print('-- Loading the config file and the dataset')
    cfg, dataset = _load_cfg_and_ds(config_file, gpu)
    
    print('-- Running the model and saving the results')
    for basin in tqdm(dataset.basins, disable=cfg .disable_pbar, file=sys.stdout):

        for period in dataset.start_and_end_dates.keys():
            
            # Extract the basin data
            if period == 'train':
                model_concept = get_concept_model(cfg, dataset.ds_train)
                basin_data = dataset.ds_train.sel(basin=basin)
            elif period == 'test':
                model_concept = get_concept_model(cfg, dataset.ds_test)                
                basin_data = dataset.ds_test.sel(basin=basin)
            elif period == 'valid':
                model_concept = get_concept_model(cfg, dataset.ds_valid)
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

# python thn_run.py train --config-file ../examples/config_run_m0.yml
if __name__ == "__main__":
    _main()