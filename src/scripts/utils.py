import yaml
import numpy as np
import os
from pathlib import Path
import pandas as pd
import sys
import matplotlib.pyplot as plt

# Make sure code directory is in path,
# Add the parent directory of your project to the Python path
project_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(project_dir)

from src.utils.metrics import metric_name_func_dict

# Expand the '~' to the full path of the home directory
home_directory = os.path.expanduser('~')

def load_config(config_file):

    # Read the config file
    if config_file.exists():
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)           
    else:
        raise FileNotFoundError(f"File not found: {config_file}")
    
    # Create the output directory
    plots_folder = cfg['plots_folder']
    os.makedirs(plots_folder, exist_ok=True)
    
    return cfg

def annotate_statistics(ax, data, statistic='mean', color='tab:red', gap=0.05, fontsize=12, **kwargs):
    
    data_aux = data[data > 0]
    
    if statistic == 'mean':
        value = np.mean(data_aux)
        label = f'Mean: {value:.3f}'
    elif statistic == 'median':
        value = np.median(data_aux)
        label = f'Median: {value:.3f}'
    else:
        raise ValueError("Invalid statistic. Choose 'mean' or 'median'.")

    ax.axvline(value, color=color, linestyle='--', linewidth=2)
    
    # Calculate the gap
    ylim = ax.get_ylim()
    gap_height = gap * (ylim[1] - ylim[0])
    
    if statistic == 'mean':
        # Check if mean is greater (right) or smaller (left) than the median
        if value > np.median(data_aux):
            ax.text(value + 100*gap, ylim[1] - gap_height, label, va='top', ha='left', color=color, fontsize=fontsize, **kwargs)
        else:
            ax.text(value - gap, ylim[1] - gap_height, label, va='top', ha='right', color=color, fontsize=fontsize, **kwargs)
    elif statistic == 'median':
        # Check if median is greater (right) or smaller (left) than the mean
        if value > np.mean(data_aux):
            ax.text(value + gap, ylim[1] - gap_height, label, va='top', ha='left', color=color, fontsize=fontsize, **kwargs)
        else:
            ax.text(value - 100*gap, ylim[1] - gap_height, label, va='top', ha='right', color=color, fontsize=fontsize, **kwargs)

def load_results_path(results_folder: Path, periods: list):
    '''
    Load the path to the model_metrics folder within the results_folder.
    If the model_metrics folder does not exist, it will be created by merging the metrics from the different folders.

    - Args:
        results_folder (Path): Path to the folder containing the results.
        periods (list): List of periods for which the metrics are calculated.
        
    - Returns:
        model_metrics_path (Path): Path to the model_metrics folder.
        results_folder (Path): Path to the results folder.
    '''

    # Check if '~' is in the path and expand it to the full path of the home directory
    if '~' in str(results_folder):
        results_folder = Path(str(results_folder).replace('~', home_directory))

    if results_folder.exists():

        # Check if it contains a folder named 'model_metrics'
        model_metrics_path = results_folder / 'model_metrics'
        if model_metrics_path.exists():
            return model_metrics_path, results_folder
        else:
            # Create a folder named 'model_metrics'
            os.makedirs(model_metrics_path, exist_ok=True)

            # There might be a bunch of single folders that have to be merged
            # Get list of folders
            folders = sorted([f.name for f in results_folder.iterdir() if f.is_dir()])
            # Iterate over each period and folder
            for period in periods:
                df_period = pd.DataFrame()
                for folder in folders:
                    # Ensure that folder is a Path object
                    folder_path = results_folder / folder
                    # Construct the path for the model_metrics directory within each folder
                    metrics_folder = folder_path / 'model_metrics'
                    # Find a file that contains 'metrics_period' in its name within the folder / model_metrics
                    for file in metrics_folder.glob(f'*metrics_{period}*'):
                        if file.is_file():
                            # Load the file
                            df = pd.read_csv(file)
                            # Add to df_period
                            df_period = pd.concat([df_period, df], ignore_index=True)
                
                # Save the merged dataframe
                df_period.to_csv(model_metrics_path / f'metrics_{period}.csv', index=False)

            return model_metrics_path, results_folder

    else:
        raise FileNotFoundError(f"Folder not found: {results_folder}")

def metrics_from_julia_results(metrics, period, results_path, metrics_path):

    '''
    Load the results from the Julia model and calculate the metrics for each basin.
    
    - Args:
        metrics (list): List of metrics to calculate.
        period (str): Period for which the metrics are calculated.
        results_path (Path): Path to the folder containing the results.
        metrics_path (Path): Path to the folder where the metrics will be saved.
        
    - Returns:
        df_metrics (DataFrame): DataFrame containing the metrics for each basin.
    '''

    # Check if 'model_results' folder exists in the results_path
    if 'model_results' in [f.name for f in results_path.iterdir() if f.is_dir()]:
        model_results_path = results_path / 'model_results'

        # List all files in the model_results folder that contain the period
        files = sorted([str(f) for f in model_results_path.glob(f'*{period}*') if f.is_file()])

        # Create a dataframe to store the metrics (columns: basin_id, metric1, metric2, ...)
        df_metrics = pd.DataFrame(columns=['basin_id'] + metrics)
        # Load the results for each file
        for file in files:
            df = pd.read_csv(file)
            # Filter by columns 'Date', 'q_bucket', and 'q_obs' or 'y_sim' and 'y_obs'
            if 'y_sim' in df.columns:
                # Rename the columns
                df = df.rename(columns={'y_sim': 'q_bucket', 'y_obs': 'q_obs'})
            
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'date'})

            df = df[['date', 'q_bucket', 'q_obs']]
            # Extract basin name from the file name
            basin_id = file.split('/')[-1].split('_')[0]
            # Calculate the metrics for the basin and add
            basin_metrics = [metric_name_func_dict[metric](df['q_obs'].values, df['q_bucket'].values) for metric in metrics]

            # Add the basin_id and the metrics to the dataframe
            df_metrics.loc[len(df_metrics)] = [basin_id] + basin_metrics

        # Save the dataframe to a csv file
        df_metrics.to_csv(metrics_path / f'metrics_{period}.csv', index=False)

        return df_metrics

    else:
        raise FileNotFoundError(f"Folder not found: {results_path / 'model_results'}")

def plot_metric_histogram(df_period, metric, threshold_dict, graph_title, period, plots_folder):
    """
    Plots a histogram of the metric values from a DataFrame with optional threshold adjustments.
    
    Parameters:
    - df_period: DataFrame containing the data to be plotted.
    - metric: The metric column name to be plotted.
    - threshold_dict: Dictionary containing threshold values and types for metrics.
    - graph_title: Title for the histogram graph.
    - period: The period label to be included in the title.
    - plots_folder: Path to the folder where the plot will be saved.
    - annotate_statistics: Function to annotate the plot with statistical values.
    
    Returns:
    - None. The function plots and saves the histogram.
    """

    # Identify the basin column - column with 'basin' in its name
    basin_column = [col for col in df_period.columns if 'basin' in col.lower()]
    # Rename the column to 'basin' if it is not already named 'basin'
    if len(basin_column) == 1:
        df_period.rename(columns={basin_column[0]: 'basin'}, inplace=True)
    else:
        raise ValueError('Column containing basin ID not found in the dataframe (at least one column should contain "basin" in its name)')

    # Filter by columns basin_ID and metrics
    df = df_period[['basin', metric]].copy()  # Explicitly create a copy

    # Plot histogram of the metric values
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if metric in threshold_dict:
        th_value = threshold_dict[metric][0]
        th_type = threshold_dict[metric][1]

        if th_type == 'greater':
            # Count basins below the threshold
            n_below_threshold = (df[metric] <= th_value).sum()
            # Values below the threshold to be equal to the threshold
            df.loc[:, metric] = np.where(df[metric] <= th_value, th_value, df[metric])

            # Clean nan values - replace with threshold
            df.loc[:, metric] = df[metric].fillna(th_value)

        else:
            # Count basins below the threshold
            n_below_threshold = (df[metric] >= th_value).sum()
            # Values below the threshold to be equal to the threshold
            df.loc[:, metric] = np.where(df[metric] >= th_value, th_value, df[metric])
            # Clean nan values - drop rows with nan values
            df = df.dropna(subset=[metric])

    # Calculate the bins for the histogram
    hist_values = df[metric].values
    n_bins = 20
    min_value = hist_values.min()
    max_value = hist_values.max()
    bins = np.linspace(min_value, max_value, n_bins + 1)

    # Plot histogram
    ax.hist(hist_values, bins=bins, color='tab:blue', alpha=0.5)

    # Add mean and median value plot to the histogram
    annotate_statistics(ax, hist_values, statistic='mean', color='tab:red', gap=0.01, fontsize=12)
    annotate_statistics(ax, hist_values, statistic='median', color='green', gap=0.01, fontsize=12)

    # Add text in the top left corner with the number of basins
    n_basins = df['basin'].nunique()
    ax.text(0.02, 0.98, f'{n_basins} basins', transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=12, color='black')

    if metric in threshold_dict:
        ax.set_title(f'{graph_title} (${metric.upper()} \leq {th_value}$:  {n_below_threshold} counts) | {period} period')
    else:
        ax.set_title(f'{graph_title} | {period} period')
    ax.set_xlabel(f'{metric.upper()}')
    ax.set_ylabel('Frequency')
    plt.show()

    # Save the plot
    fig.savefig(plots_folder / f'{metric}_histograms_{period}.png', dpi=150, bbox_inches='tight')

# Example usage:
# plot_metric_histogram(df, 'accuracy', threshold_dict, 'Accuracy Histogram', '2023', Path('./plots'), annotate_statistics)
    

if __name__ == '__main__':
    print('This script is not meant to be run directly.')
    exit(1)