import sys
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
import math
import itertools
from datetime import datetime
import torch

project_dir = str(Path.cwd().parent.parent.parent)  # Adjust parent levels as needed
sys.path.append(project_dir)

from src.utils.plots import (
    get_reprojected_coords,
    ORIGINAL_CRS,
    TARGET_CRS
)

import random
#Set the seed for reproducibility
random.seed(42)

# Define custom colors for clusters to match the image
cluster_colors = {
    0: '#1f77b4',  # Blue
    1: '#ff7f0e',  # Orange
    2: '#2ca02c',  # Green
    3: '#d62728',  # Red
    4: '#9467bd',  # Purple
    5: '#8c564b',  # Brown
    6: '#e377c2',  # Pink
    7: '#7f7f7f',  # Gray
    8: '#bcbd22',  # Yellow
    9: '#17becf',  # Cyan
    10: '#ff0000',  # Red
    11: '#00ff00',  # Green
    12: '#0000ff',  # Blue
    13: '#ffff00',  # Yellow
    14: '#ff00ff',  # Magenta
    15: '#00ffff',  # Cyan
    16: '#000000',  # Black
    17: '#ffffff',  # White
    18: '#800000',  # Maroon
    19: '#808000',  # Olive
    20: '#008000',  # Green
    21: '#800080',  # Purple
    22: '#008080',  # Teal
    23: '#000080',  # Navy
    24: '#808080',  # Gray
    25: '#c0c0c0',  # Silver
}

# Light gray color for non-selected basins
non_selected_color = '#d3d3d3'

def random_basins_subset(cluster_files, fraction):
    """
    Randomly selects a fraction of basins from each cluster file and saves the selected basins in a CSV format.

    Parameters:
    cluster_files (list of PosixPath): List of file paths for the basin cluster files.
    fraction (float): The fraction of basins to randomly select from each file.
    
    Returns:
    selected_basins_dict (dict): A dictionary where the keys are the cluster names and the values are the selected basins.
    non_selected_basins_dict (dict): A dictionary where the keys are the cluster names and the values are the non-selected basins.
    output_file (str): The name of the output file where the selected basins by cluster are saved.
    basin_file (str): The name of the output file where the list of selected basins is saved.
    """
    selected_basins_dict = {}
    non_selected_basins_dict = {}

    for icluster, cluster_file in enumerate(cluster_files):
        # Read the basins from the file
        with cluster_file.open('r') as file:
            basins = [line.strip() for line in file.readlines()]
        
        # Calculate the number of basins to select, rounding up
        num_to_select = math.ceil(len(basins) * fraction)
        
        # Randomly select the basins
        selected_basins = random.sample(basins, num_to_select)
        
        # Determine the non-selected basins
        non_selected_basins = list(set(basins) - set(selected_basins))
        
        # Store the results for this cluster
        # result[f'cluster{icluster+1}'] = (selected_basins, non_selected_basins)
        selected_basins_dict[f'cluster{icluster+1}'] = selected_basins
        non_selected_basins_dict[f'cluster{icluster+1}'] = non_selected_basins
    
    # Write the selected basins to the output file
    # Fraction is multiplied by 100 to convert it to a percentage
    fraction = int(fraction * 100)
    output_file = f'random_sample_{len(cluster_files)}clusters_{fraction}percent.csv'
    with open(output_file, 'w') as f_out:
        for cluster_name, selected_basins in selected_basins_dict.items():
            # Write the cluster name followed by the selected basins, separated by commas
            f_out.write(f"{cluster_name}," + ",".join(selected_basins) + "\n")

    # List of selected basins
    selected_basins = sorted([basin for basins in selected_basins_dict.values() for basin in basins])
    # Save the list of selected basins to a text file
    # 569_basin_file_cluster1of6_63
    basin_file = f'{len(selected_basins)}_basin_file_sample.txt'
    with open(basin_file, 'w') as f_out:
        for basin in selected_basins:
            f_out.write(f"{basin}\n")

    
    
    return selected_basins_dict, non_selected_basins_dict, output_file, basin_file

def plot_basin_sample(selected_basins_dict, non_selected_basins_dict, 
                         hm_catchment_path, map_shape_path):
    
    # Load and reproject the shapefiles
    states, hm_catchment_gdf = get_reprojected_coords(hm_catchment_path, map_shape_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the background map (e.g., states)
    states.plot(ax=ax, facecolor='#f0f0f0', edgecolor='black', linewidth=0.5)

    # Values from Str to int
    selected_basins_dict = {cluster: [int(basin) for basin in basins] for cluster, basins in selected_basins_dict.items()}
    non_selected_basins_dict = {cluster: [int(basin) for basin in basins] for cluster, basins in non_selected_basins_dict.items()}
    num_clusters = len(selected_basins_dict)
    # cluster_counts is total number of basins in each cluster
    cluster_counts = [len(selected_basins_dict[cluster]) + len(non_selected_basins_dict[cluster]) for cluster in selected_basins_dict.keys()]
    selected_counts = [len(selected_basins_dict[cluster]) for cluster in selected_basins_dict.keys()]

    # Plot non-selected basins with a fixed color
    for non_selected_basins in non_selected_basins_dict.values():
        non_selected_gdf = hm_catchment_gdf[hm_catchment_gdf['hru_id'].isin(non_selected_basins)]
        
        if not non_selected_gdf.empty:
            # Create a GeoDataFrame of basin coordinates (lon_cen, lat_cen)
            basin_coords = gpd.points_from_xy(non_selected_gdf['lon_cen'], non_selected_gdf['lat_cen'], crs=ORIGINAL_CRS)
            basin_coords = gpd.GeoDataFrame(geometry=basin_coords, crs=ORIGINAL_CRS)
            
            # Reproject to the target CRS
            basin_coords = basin_coords.to_crs(TARGET_CRS)
            
            # Plot non-selected basins as circles
            ax.scatter(basin_coords.geometry.x, basin_coords.geometry.y, 
                    s=50, color=non_selected_color, edgecolor='black', alpha=0.5, label='Non-selected')


    # Plot selected basins for each cluster with different colors
    for icluster, selected_basins in enumerate(selected_basins_dict.values()):
        selected_gdf = hm_catchment_gdf[hm_catchment_gdf['hru_id'].isin(selected_basins)]
        
        if not selected_gdf.empty:
            # Create a GeoDataFrame of basin coordinates (lon_cen, lat_cen)
            basin_coords = gpd.points_from_xy(selected_gdf['lon_cen'], selected_gdf['lat_cen'], crs=ORIGINAL_CRS)
            basin_coords = gpd.GeoDataFrame(geometry=basin_coords, crs=ORIGINAL_CRS)
            
            # Reproject to the target CRS
            basin_coords = basin_coords.to_crs(TARGET_CRS)
            
            # Plot selected basins as circles
            ax.scatter(basin_coords.geometry.x, basin_coords.geometry.y, 
                    s=50, color=cluster_colors[icluster], edgecolor='black')


    # Calculate the percentage of selected basins
    num_selected = sum(len(basins) for basins in selected_basins_dict.values())
    num_total = sum(len(basins) for basins in selected_basins_dict.values()) + sum(len(basins) for basins in non_selected_basins_dict.values())
    percent_selected = num_selected / num_total * 100

    # Add custom legend with circular markers
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
            markeredgecolor='k', markeredgewidth=1, label=f'Cluster {i+1} ({selected_counts[i]}/{cluster_counts[i]})')
        for i, color in cluster_colors.items() if i < num_clusters
    ]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.12, 0.95))

    # Customize plot appearance
    ax.set_title(f"Random Sample of Basins | {percent_selected:.0f}% selected by cluster")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Get rid of box lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Save the plot
    plt.tight_layout()
    file_name = f'random_sample_{num_clusters}clusters_{int(percent_selected)}percent.png'
    plt.savefig(file_name, dpi=150, bbox_inches='tight')

    # Display the plot
    plt.show()

def load_hyperparameters(file='hyperparameters.yml'):
    """
    Load the hyperparameters from the YAML file.

    Parameters:
    file (str): The name of the YAML file containing the hyperparameters.

    Returns:
    hyperparameters (dict): A dictionary containing the hyperparameters.
    """
    import yaml

    with open(file, 'r') as f:
        hyperparameters = yaml.safe_load(f)

    return hyperparameters

def create_finetune_folder(base_name='runs_finetune'):
    """
    Creates a directory with an incrementing suffix if the directory already exists.
    
    Parameters:
    base_name (str): The base name for the folder. Defaults to 'runs_finetune'.
    
    Returns:
    Path: The path to the newly created directory.
    """

    now = datetime.now()
    dt_string = now.strftime("%y%m%d_%H%M%S")

    # counter = 1
    # while True:
    #     finetune_folder = Path(f'{base_name}_{counter}')
    #     if not finetune_folder.exists():
    #         finetune_folder.mkdir(exist_ok=False)  # Create the directory
    #         return finetune_folder
    #     counter += 1

    finetune_folder = Path(f'{base_name}_{dt_string}')
    finetune_folder.mkdir(exist_ok=False)  # Create the directory
    
    return finetune_folder

def hyperparameter_combinations(hyperparameters):
    """
    Generate all possible combinations of hyperparameters.
    
    If 'odesmethod' is 'euler' or 'rk4', include 'time_step'. If 'bosh3', exclude 'time_step'.
    """
    # Separate odesmethod and other hyperparameters
    odesmethods = hyperparameters['odesmethod']
    other_params = {k: v for k, v in hyperparameters.items() if k != 'odesmethod'}

    # Store all combinations
    all_combinations = []
    
    # Generate combinations for each odesmethod
    for method, method_config in odesmethods.items():
        # Create a base dictionary for this odesmethod
        base_params = {'odesmethod': method}
        
        # If 'time_step' exists in the method_config, include it
        if 'time_step' in method_config and method_config['time_step'] is not None:
            params_to_combine = {**other_params, 'time_step': method_config['time_step']}
        else:
            params_to_combine = other_params
        
        # Get all possible combinations for the current set of parameters
        param_names = list(params_to_combine.keys())
        param_values = list(params_to_combine.values())
        
        for combination in itertools.product(*param_values):
            combined_params = dict(zip(param_names, combination))
            combined_params.update(base_params)
            all_combinations.append(combined_params)
    
    return all_combinations

def b2mb(x): return int(x / 2**20)

class TorchTracemalloc:
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()  # Memory at the start
        torch.cuda.reset_peak_memory_stats()  # Reset the peak memory tracker
        return self

    def __exit__(self, *exc):
        self.end = torch.cuda.memory_allocated()  # Memory at the end
        self.peak = torch.cuda.max_memory_allocated()  # Peak memory during the block
        self.used = b2mb(self.end - self.begin)  # Used memory during the block
        self.peaked = b2mb(self.peak - self.begin)  # Peak memory change


if __name__ == '__main__':
    print('This script is not meant to be run directly.')
    exit(1)