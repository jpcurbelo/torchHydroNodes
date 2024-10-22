import os
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import pandas as pd
import numpy as np
import torch

from src.utils.metrics import (
    NSE_eval,
    compute_all_metrics,
)



def save_and_plot_simulation(ds, q_bucket, basin, period='train', 
                         model_name='exphydro',
                         plots_dir=None,
                         plot_prcp=False):
    '''
    Plot the model simulations and observed values for a basin and save the plot.
    
    - Args:
        ds: xarray.Dataset, dataset with the input data.
        q_bucket: array_like, model predictions.
        basin: str, basin name.
        period: str, period of the run ('train', 'test', 'valid').
        plot_prcp: bool, whether to plot the precipitation rate.
        
    '''
    
    dates = ds['date'].values
    q_obs = ds['obs_runoff'].values            
        
    # Plot the simulated and actual values
    _, ax1 = plt.subplots(figsize=(16, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Period')
    ax1.set_ylabel('Discharge (mm/day)', color=color)
    ax1.plot(dates, q_obs, label='Observed', linewidth=3, color=color, zorder=2)
    ax1.plot(dates, q_bucket, ':', linewidth=2, label='Simulated', color='tab:red', zorder=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Set the major and minor locators and formatters for the x-axis
    years = YearLocator()   # every year
    months = MonthLocator()  # every month
    yearsFmt = DateFormatter('%Y')

    # Set the x-axis locators and formatters
    ax1.xaxis.set_major_locator(years)
    ax1.xaxis.set_major_formatter(yearsFmt)
    ax1.xaxis.set_minor_locator(months)
    
    # Set the x-axis limits
    start_date = dates.min()
    end_date = dates.max()
    ax1.set_xlim(start_date, end_date)
    # Enable autoscaling for the view
    ax1.autoscale_view()
        
    # Legend
    ax1.legend(loc='upper right')
    
    if plot_prcp:
        prcp = ds['prcp(mm/day)']
        # Create a twin Axes sharing the x-axis
        ax2 = ax1.twinx()
        color = 'lightslategray'
        ax2.set_ylabel('Precipitation Rate', color=color)
        ax2.plot(dates, prcp, label='Second Y-axis Data', color=color, zorder=1)
        ax2.tick_params(axis='y', labelcolor='darkslategray')
        ax2.invert_yaxis()  # Invert the y-axis of the second axis
            
    nse_val = NSE_eval(q_obs, q_bucket)
    plt.title(f'Model Predictions ({model_name}) | $NSE = {nse_val:.3f}$ | {period} period')
    
    plot_file_name = f'{basin}_{period.lower()}.png'
    
    plt.savefig(os.path.join(plots_dir, plot_file_name), bbox_inches='tight')
    plt.close()

def compute_and_save_metrics(metrics, run_dir):
    
    metrics_dir = run_dir / 'model_metrics'
    if not metrics_dir.exists():
        metrics_dir.mkdir()
        
    # Load the results for each basin and period from the run directory / model results
    results_dir = run_dir / 'model_results'

    if not results_dir.exists():
        print(f'No results found in {results_dir}')
        return

    results_files = os.listdir(results_dir)
    # Get basins and periods
    basins = set([file.split('_')[0] for file in results_files])
    periods = set([file.split('.')[0].split('_')[-1] for file in results_files])
    
    for period in periods:
        metrics_dict = dict()
        for basin in basins:
            # Load the results from the file {basin}_results_{period}.csv
            results = pd.read_csv(results_dir / f'{basin}_results_{period}.csv')
            
            # Last column is the observed runoff
            q_obs = results.iloc[:, -1].values
            # Before the last column are the model predictions
            q_sim = results.iloc[:, -2].values
            # Dates
            dates = np.array(pd.to_datetime(results['date'].values))
            
            # Compute the metrics
            metrics_dict[basin] = compute_all_metrics(q_obs, q_sim, dates, metrics)
            
        # Save the metrics for the period
        metrics_df = pd.DataFrame(metrics_dict).T.rename_axis("basin_ID")
        
        # Sort by basin ID
        metrics_df = metrics_df.sort_index()
        
        metrics_df.to_csv(metrics_dir / f'metrics_{period}.csv', float_format='%.4e')

def log_gpu_memory_summary(device, log_file_path="gpu_memory_log.txt", abbreviated=False):
    """
    Logs the GPU memory summary to a specified file.

    Args:
        device (str): The device for which to generate the memory summary (e.g., 'cuda:0').
        log_file_path (str): The path of the file to save the memory summary.
        abbreviated (bool): Whether to abbreviate the memory summary (default is False).
    """
    # Get the memory summary as a string
    memory_summary = torch.cuda.memory_summary(device, abbreviated=abbreviated)

    # Open the file and write the memory summary
    with open(log_file_path, "w") as file:
        file.write(memory_summary)

if __name__ == "__main__":
    pass