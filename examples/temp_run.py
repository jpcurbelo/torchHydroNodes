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
from src.utils.log_results import save_plot_simulation

from src.datasetzoo import get_dataset
from src.modelzoo_concept import get_concept_model

def main(config_file):
    
    # Create a Config object for the the run config
    cfg = Config(Path(config_file))
    
    # Load the forcing and target data 
    ds = get_dataset(cfg=cfg, is_train=True, scaler=dict()) 
        
    print('-- Running the model and saving the results')
    for basin in tqdm(ds.basins[:2], disable=cfg .disable_pbar, file=sys.stdout):
        
        for period in ds.start_and_end_dates.keys():
            
            # Extract the basin data
            if period == 'train':
                model_concept = get_concept_model(cfg, ds.xr_train)
                basin_data = ds.xr_train.sel(basin=basin)
            elif period == 'test':
                model_concept = get_concept_model(cfg, ds.xr_test)
                basin_data = ds.xr_test.sel(basin=basin)
            elif period == 'valid':
                model_concept = get_concept_model(cfg, ds.xr_valid)
                basin_data = ds.xr_valid.sel(basin=basin)
            else:
                raise ValueError("Invalid period. Please specify 'train', 'test', or 'valid'.")

            # Run the model
            s_snow, s_water, q_bucket,\
            et_bucket, m_bucket, ps_bucket, pr_bucket = model_concept.run(basin=basin)


            
            # Save the results
            model_concept.save_results(basin_data, 
                                    (s_snow, s_water, q_bucket, et_bucket, m_bucket, ps_bucket, pr_bucket), 
                                    basin,
                                    period=period
                                    )
            
            # Plot the results 
            save_plot_simulation(ds=basin_data,
                                q_bucket=q_bucket,
                                basin=basin,
                                period=period,
                                model_name=cfg.concept_model,
                                plots_dir=cfg.plots_dir,
                                plot_prcp=False
                            )
            
            
            # aux = input("Press Enter to continue...")
    

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Run file to test temporary code')
    parser.add_argument('--config-file', type=str, default='config_run.yml', help='Path to the config file')
    args = parser.parse_args()
    
    
    main(args.config_file)