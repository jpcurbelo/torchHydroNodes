from pathlib import Path
import yaml
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cmcrameri import cm
from matplotlib.lines import Line2D
import re
import numpy as np

# Get the current working directory (works for Jupyter or interactive environments)
project_dir = str(Path.cwd().parent.parent.parent)  # Adjust parent levels as needed
sys.path.append(project_dir)

from src.utils.plots import (
    plot_histograms_period,
    apply_threshold,
)

# Configuration file path
# COMBO_FILE = Path('config_file_process_combos_fract01.yml')
# COMBO_FILE = Path('config_file_process_combos_fract02.yml')
# COMBO_FILE = Path('config_file_process_combos_fract01_euler05d.yml')
COMBO_FILE = Path('config_file_process_combos_fract01_seeds.yml')

ONLY_PLOT = 0   # True or False
SEEDS_RUN = 1   # True or False


def load_combinations(file_path):
    """Load the combinations configuration file."""
    with open(file_path, 'r') as f:
        combinations = yaml.safe_load(f)
    return combinations

def load_config_value(combo_folder_path, key):
    """
    Load a specific value from the config_com*.yml file based on the provided key.
    """
    config_file = next(combo_folder_path.glob('config_com*.yml'), None)
    if config_file is None:
        raise FileNotFoundError(f"No config_comX.yml found in {combo_folder_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get(key, '')
    
def is_basin_completed(basin_folder):
    """
    Check if a basin is completed by verifying the presence of the 'evaluation' step
    in the 'epoch_stats.csv' file and compute mean time/memory if completed.
    """
    epoch_stats_file = basin_folder / 'epoch_stats.csv'
    
    # Check if the file exists
    if not epoch_stats_file.exists():
        # print(f"No epoch_stats.csv found in {basin_folder}")
        return False, None

    # Read the CSV and check if the 'evaluation' step is present
    try:
        df = pd.read_csv(epoch_stats_file)
        # We assume the 'epoch' column contains either numeric values or 'evaluation'

        if 'evaluation' in df['epoch'].values:
            # Basin is completed, compute mean stats
            mean_time, mean_memory = compute_mean_stats(df)
            return True, (mean_time, mean_memory)
        else:
            # print(f"'evaluation' step not found in {epoch_stats_file}")
            return False, None
    except Exception as e:
        # print(f"Error reading {epoch_stats_file}: {e}")
        return False, None

def compute_mean_stats(df):
    """
    Compute the mean time per epoch and mean memory peak per epoch for the basin.
    """

    # Filter out non-numeric rows such as 'evaluation' and 'final_plot'
    numeric_epochs = df[pd.to_numeric(df['epoch'], errors='coerce').notna()]

    # Remove any rows with 'N/A' in the columns we are interested in
    numeric_epochs = numeric_epochs.dropna(subset=['epoch_time_seg', 'cpu_peak_memory_mb'])
    
    # Compute mean time per epoch and mean memory peak per epoch
    mean_time = numeric_epochs['epoch_time_seg'].mean()
    mean_memory = numeric_epochs['cpu_peak_memory_mb'].mean()

    return mean_time, mean_memory

def process_combination_folder(combo_folder_path):
    """
    Process each combination folder, check if the basin is completed, and compute stats.
    """

    experiment_name = load_config_value(combo_folder_path, 'experiment_name')

    # Iterate over the basins inside the combination folder
    basin_folders = sorted([f for f in combo_folder_path.iterdir() if f.is_dir() and f.name.startswith(experiment_name)])

    completed_basins = []
    basin_stats = {}  # To store mean time and memory for each basin
    
    for basin_folder in basin_folders:
        is_completed, stats = is_basin_completed(basin_folder)
        if is_completed:
            completed_basins.append(basin_folder.name)
            mean_time, mean_memory = stats
            # Save the computed stats for this basin
            basin_stats[basin_folder.name] = {'mean_time': mean_time, 'mean_memory': mean_memory}
            # print(f"Basin {basin_folder.name}: Mean time/epoch = {mean_time}, Mean memory peak/epoch = {mean_memory}")

    # Sort the completed basins
    completed_basins.sort()
    # Sort the basin stats by basin ID
    basin_stats = dict(sorted(basin_stats.items()))
    
    # Return completed basins and total number of basins
    return completed_basins, len(basin_folders), basin_stats

def save_basin_stats(combo_folder_path, basin_stats, periods=['valid']):
    
    # Create folder combination_results if it does not exist
    output_folder = combo_folder_path / 'combo_results'
    output_folder.mkdir(parents=True, exist_ok=True)

    experiment_name = load_config_value(combo_folder_path, 'experiment_name')

    for period in periods:

        output_file = output_folder / f'combo_stats_{period}.csv'

        # Load metrics for the current period (valid or train)
        run_metrics = load_config_value(combo_folder_path, 'metrics')
        # run_metrics to lower case
        run_metrics = [metric.lower() for metric in run_metrics]
        fieldnames = ['basin', 'epoch_time_seg_mean', 'cpu_peak_memory_mb_mean'] + run_metrics

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for basin, stats in basin_stats.items():
                # Extract basin ID (8 characters after experiment name)
                basin_id_start = len(experiment_name)
                basin_id = basin[basin_id_start:basin_id_start + 8].strip('_')
                
                # Round numeric values to 2 decimal places
                rounded_time = round(stats['mean_time'], 2)
                rounded_memory = round(stats['mean_memory'], 2)

                # Load metrics for the current period (valid or train)
                metrics_data = load_metrics_period(Path(combo_folder_path / basin), period)

                # Round each metric value to 5 decimal places
                for key, value in metrics_data.items():
                    if isinstance(value, (int, float)):
                        metrics_data[key] = round(value, 5)

                # Create a dictionary with the data to write
                data = {'basin': basin_id, 'epoch_time_seg_mean': rounded_time, 'cpu_peak_memory_mb_mean': rounded_memory}
                data.update(metrics_data)

                writer.writerow(data)

def load_metrics_period(basin_folder, period):
    """
    Load evaluation metrics for a given period (valid or train) from the basin folder.
    """
    metrics_folder = basin_folder / 'model_metrics'
    metrics_file = metrics_folder / f'evaluation_metrics_{period}.csv'

    # Initialize an empty dictionary
    metrics_data = {}

    # Check if the metrics file exists and load it
    if metrics_file.exists():
        df = pd.read_csv(metrics_file)
        metrics_data = df.iloc[0].to_dict()  # Load the first row as a dict

    return metrics_data

# Function to extract the number at the end of the folder name
def get_combination_number(folder_name):
    match = re.search(r'run_combination(\d+)$', folder_name)
    return int(match.group(1)) if match else float('inf')

def plot_performance_scatter(main_folder, run_folders_labels, run_metrics=['nse'], periods=['valid'], 
                             threshold_dict=None, topN=5, collect_seeds_results=False):
    """
    Scatter plot with mean time and memory on the axes and NSE (or other metrics) as the size of the circle.
    """

    # Construct full paths for each folder
    run_folders_paths = {folder: f"{main_folder}/{folder}" for folder in run_folders_labels}

    for period in periods:

        times = []
        memories = []
        metric_values = {metric: [] for metric in run_metrics}  # Reset for each period
        combo_labels = []  # To store labels for each combo

        number_of_basins = 0

        if collect_seeds_results:
            # Optional: To store seed data for error bars
            seeds_metric_data = {metric: {} for metric in run_metrics}  # Dictionary to store seed values

        # Loop over the run folders and combination folders
        for folder, label in run_folders_labels.items():

            run_folder_path = Path(run_folders_paths[folder])
            # Sorting combination folders based on the numerical value at the end
            combination_folders = sorted(
                [f for f in run_folder_path.iterdir() if f.is_dir()],
                key=lambda x: get_combination_number(x.name)
            )

            icombo = 0
            for combo_folder in combination_folders:
        
                # Load the data for the given period (assuming the metrics are saved in 'combo_stats_{period}.csv')
                stats_path = combo_folder / 'combo_results' / f'combo_stats_{period}.csv'
                if not stats_path.exists():
                    print(f"No stats file found for {combo_folder} (period {period}). Skipping.")
                    continue

                icombo += 1
                df = pd.read_csv(stats_path)

                if df.empty:
                    print(f"Empty stats file found for {combo_folder} and period {period}. Skipping.")
                    continue

                if len(df) > number_of_basins:
                    number_of_basins = len(df)

                # Calculate the mean values for time, memory, and the specified metrics
                mean_time = df['epoch_time_seg_mean'].mean()
                mean_memory = df['cpu_peak_memory_mb_mean'].mean()
                
                times.append(mean_time)
                memories.append(mean_memory)

                # Store labels for each combination
                combo_labels.append(f"{label}/combo{icombo}")

                # # Calculate meadian values for each metric (e.g., 'nse', 'kge')
                # for metric in run_metrics:

                #     if metric in df.columns:
                #         df, _, _ = apply_threshold(df, metric, threshold_dict)
                #         # metric_median = df[metric].mean()
                #         metric_median = df[metric].median()
                #         metric_values[metric].append(metric_median)
                #     else:
                #         metric_values[metric].append(None)  # In case the metric isn't present

                # Define the weights for mean and median
                weight_median = 1.0   #1.0   #0.6
                weight_mean   = 0.0   #0.0   #0.4

                # Calculate combined metric for each configuration
                for metric in run_metrics:
                    if metric in df.columns:
                        df, _, _ = apply_threshold(df, metric, threshold_dict)
                        metric_mean = df[metric].mean()
                        metric_median = df[metric].median()

                        # Collect seed data if requested -> for error bars
                        if collect_seeds_results:
                            # Ensure that `seeds_metric_data[metric][label]` exists
                            if label not in seeds_metric_data[metric]:
                                seeds_metric_data[metric][label] = []
                            
                            # Append the median value (since it's a single number, not iterable)
                            seeds_metric_data[metric][label].append(metric_median)

                        # Combined metric calculation
                        combined_metric = weight_mean * metric_mean + weight_median * metric_median
                        
                        # Append the result to your metric values dictionary or list
                        metric_values[metric].append(combined_metric)
                    else:
                        # In case the metric isn't present, append None or an appropriate placeholder
                        metric_values[metric].append(None)

        # Create scatter plot based on the specified metrics (e.g., NSE for size)
        for metric in run_metrics:

            plt.figure(figsize=(12, 6))
            metric_data = metric_values[metric]

            # Filter out None and NaN values from metric_data, times, and memories for plotting
            valid_data = [(t, m, n) for t, m, n in zip(times, memories, metric_data) if not pd.isna(n)]
            if not valid_data:  # Skip if no valid data
                print(f"No valid data for {metric} in period {period}. Skipping plot.")
                continue

            times_filtered, memories_filtered, filtered_metric_data = zip(*valid_data)

            # Define the colormap and normalization based on the metric values
            cmap = cm.oslo  # You can choose any colormap from cmcrameri
            normalize = mcolors.Normalize(vmin=min(filtered_metric_data), vmax=max(filtered_metric_data))

            # Normalize sizes between min_size and max_size
            min_size = 50  # Minimum circle size
            max_size = 500  # Maximum circle size
            size_normalize = mcolors.Normalize(vmin=min(filtered_metric_data), vmax=max(filtered_metric_data))
            sizes = [min_size + (max_size - min_size) * size_normalize(n) for n in filtered_metric_data]

            # Create scatter plot with normalization
            scatter = plt.scatter(times_filtered, memories_filtered, 
                      s=sizes,  # Normalized size of the points
                      c=filtered_metric_data,  # Color of the points
                      cmap=cmap,  # Colormap for the points
                      norm=normalize,  # Normalization for the colormap
                      alpha=1.0,  # Transparency of the points
                      edgecolors='black',  # Border color
                      linewidths=1)  # Border width

            # Adding color bar for the metric
            cbar = plt.colorbar(scatter)
            cbar.set_label(f'${metric.upper()}$ (median value)', fontsize=12)  # Color bar label font size
            cbar.ax.tick_params(labelsize=12)  # Color bar tick label size

            # Add labels and title with larger font sizes
            plt.xlabel('Mean Time per Epoch ($s$)', fontsize=12)
            plt.ylabel('Mean Memory Peak ($MB$)', fontsize=12)
            plt.title(f'Time vs Memory (circle size = ${metric.upper()}$, period = {period})', fontsize=14)
            
            # Increase font size for tick labels
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # Highlight the top 5 results by putting the numbers in the center of the circles
            top_N_indices = sorted(range(len(filtered_metric_data)), 
                                   key=lambda i: filtered_metric_data[i], reverse=True)[:topN]

            legend_circles = []
            top_N_labels = []
            for idx, rank in zip(top_N_indices, range(1, topN+1)):
                font_size = sizes[idx] / 30
                plt.text(times_filtered[idx], memories_filtered[idx], str(rank), fontsize=font_size, 
                        ha='center', va='center', color='black', fontweight='bold')
                
                circle = Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=cmap(normalize(filtered_metric_data[idx])), 
                                markersize=10, markeredgewidth=1.5, markeredgecolor='black')
                legend_circles.append(circle)
                top_N_labels.append(f"{rank}-{combo_labels[idx]} | ${metric.upper()}$ = {filtered_metric_data[idx]:.3f}")
            
            # Adding legend for the colormap
            # plt.legend(legend_circles, top_N_labels, fontsize=9)
            legend = plt.legend(legend_circles, top_N_labels, fontsize=9, loc='upper left', bbox_to_anchor=(1.2, 1.0))
            legend.set_title(f"Subsampled basins: {number_of_basins}", prop={'size': 9, 'weight': 'bold'})

            # # Format the ticks in scientific notation
            # plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

            # Log scale for memory axis
            plt.yscale('log')

            plt.tight_layout()

            # Save the plot
            # plt.savefig(f'performance_{metric}_{period}.png')
            plt.savefig(Path(main_folder) / f'performance_{metric}_{period}.png', bbox_inches='tight', dpi=150)

            # Close the plot to avoid memory issues
            plt.close()


            # Optional: Plot error bars for seeds data
            if collect_seeds_results:
                # Plot error bars for seeds data
                plt.figure(figsize=(12, 6))

                combo_means = []
                combo_stds = []
                combo_mins = []  # To store minimum values for dashed lines
                combo_labels_sorted = []
                seeds_count = []

                for combo_label, seed_values in seeds_metric_data[metric].items():
                    if seed_values:
                        combo_means.append(np.mean(seed_values))
                        combo_stds.append(np.std(seed_values))  # Or use other variability measures
                        combo_mins.append(np.min(seed_values))  # Collect minimum value across seeds
                        combo_labels_sorted.append(combo_label)
                        seeds_count.append(len(seed_values))  # Collect the number of seeds for each combo

                # Sort by decreasing mean
                sorted_indices = np.argsort(combo_means)[::-1]
                combo_means = np.array(combo_means)[sorted_indices]
                combo_stds = np.array(combo_stds)[sorted_indices]
                combo_mins = np.array(combo_mins)[sorted_indices]  # Sort minimum values accordingly
                combo_labels_sorted = np.array(combo_labels_sorted)[sorted_indices]

                # Create a bar plot with error bars
                plt.bar(combo_labels_sorted, combo_means, yerr=combo_stds, capsize=5, alpha=0.75)

                # Draw dashed lines for minimum values
                for i in range(len(combo_means)):
                    min_value = combo_mins[i]
                    plt.hlines(min_value, i - 0.4, i + 0.4, colors='red', linestyles='dashed', linewidth=2)

                # Angle x-labels and adjust font size
                plt.xticks(rotation=45, ha='right', fontsize=9)

                # Add ylabel and title, with number of seeds in title
                max_seeds = max(seeds_count)  # Maximum number of seeds across all combos
                plt.ylabel(f'{metric.upper()} (Mean ± Std)')
                plt.title(f'{metric.upper()} Across Configurations (with {max_seeds} Seeds)')

                # Adjust the scale to highlight std deviations better
                plt.ylim([max(0, min(combo_means) - 2*max(combo_stds)), max(combo_means) + 1.2*max(combo_stds)])

                plt.tight_layout()

                # Save or show the error bar plot
                plt.savefig(Path(main_folder) / f'error_bars_{metric}_{period}.png', bbox_inches='tight', dpi=150)
                plt.close()



            
##########################################################################################################
def main(combo_file=COMBO_FILE):
    # Load combinations from the YAML file
    combinations = load_combinations(combo_file)

    # Accessing the main folder and other parameters
    main_folder = combinations['main_folder']
    run_folders_labels = combinations['run_folders']
    # Construct full paths for each folder
    run_folders_paths = {folder: f"{main_folder}/{folder}" for folder in run_folders_labels}
    periods = combinations['periods']
    run_metrics = combinations['metrics']
    threshold_dict = combinations['threshold_dict']
    
    if not ONLY_PLOT:
        # Iterate through run folders
        for folder in run_folders_labels.keys():

            run_folder_path = Path(run_folders_paths[folder])
            # Sorting combination folders based on the numerical value at the end
            combination_folders = sorted(
                [f for f in run_folder_path.iterdir() if f.is_dir()],
                key=lambda x: get_combination_number(x.name)
            )
        
            # Process each run folder
            for icombo, combo_folder in enumerate(combination_folders):

                completed_basins, total_basins, basin_stats = process_combination_folder(combo_folder)

                if len(completed_basins) < total_basins:
                        print(f"- {combo_folder} | {len(completed_basins)}/{total_basins} basins completed.")

                save_basin_stats(combo_folder, basin_stats, periods)

                # Plot histograms for the completed basins
                epochs = load_config_value(combo_folder, 'epochs')
                plot_histograms_period(
                    combo_folder / 'combo_results',
                    periods,
                    run_metrics,
                    threshold_dict,
                    f'Combo {icombo+1}',
                    epochs,
                    combo_folder / 'combo_results',
                    metric_base_fname='combo_stats'
                )

    # Generate plots
    plot_performance_scatter(main_folder, run_folders_labels, run_metrics, periods, 
                             threshold_dict, topN=20, collect_seeds_results=SEEDS_RUN)

if __name__ == "__main__":
    main()
