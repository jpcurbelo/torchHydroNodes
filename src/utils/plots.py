import os
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
from cmcrameri import cm

# Expand the '~' to the full path of the home directory
home_directory = os.path.expanduser('~')

# Define the CRS of your original data (if it's not already defined)
ORIGINAL_CRS = 'EPSG:4326'  # Assuming WGS84 geographic CRS
# Define the new CRS you want to reproject to, for example, an Albers Equal Area projection
TARGET_CRS = 'ESRI:102008'  # ESRI:102008 is the WKID for the Albers Equal Area projection

def get_cluster_files(cluster_folder='569_basins_6clusters'):
    '''
    Get the list of cluster files for the 569 basins with 6 clusters.
    '''

    cluster_files = sorted(list(Path(f'../../../examples/cluster_files/{cluster_folder}').glob('*.txt')))
    return cluster_files

def load_config_file(config_file, create_plots_folder=True):
    '''
    Load the configuration file and create the output directory if it does not exist.
    '''

    # Read the config file
    if config_file.exists():
        with open(config_file, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)           
    else:
        raise FileNotFoundError(f"File not found: {config_file}")
    
    # Create the output directory
    if create_plots_folder:
        plots_folder = cfg['plots_folder']
        os.makedirs(plots_folder, exist_ok=True)
    
    return cfg

def merge_model_metrics(results_folder: Path, model_metrics_path: Path, periods: list):
    '''
    Merge the model metrics for the specified periods from the different folders in the results folder.
    Save the merged metrics for each period as a CSV file in the model_metrics folder.

    - Args:
        results_folder (Path): Path to the folder containing the results.
        model_metrics_path (Path): Path to the model_metrics folder.
        periods (list): List of periods for which the metrics are calculated.
    '''

    # Create the 'model_metrics' directory if it doesn't exist
    os.makedirs(model_metrics_path, exist_ok=True)

    # Get the list of subdirectories inside the results folder
    folders = sorted([f.name for f in results_folder.iterdir() if f.is_dir()])

    # Iterate over each specified period
    for period in periods:
        df_period = pd.DataFrame()  # Initialize an empty DataFrame for the period

        # Iterate over each folder found in results_folder
        for folder in folders:
            folder_path = results_folder / folder  # Construct full folder path
            metrics_folder = folder_path / 'model_metrics'  # Subdirectory containing metrics

            # Find files matching the pattern *metrics_{period}* in the 'model_metrics' folder
            for file in metrics_folder.glob(f'*metrics_{period}*'):
                if file.is_file():
                    # Read CSV and append its content to df_period
                    df = pd.read_csv(file)
                    df_period = pd.concat([df_period, df], ignore_index=True)

        # Save the merged DataFrame for the current period as a CSV file
        output_file = model_metrics_path / f'metrics_{period}.csv'
        df_period.to_csv(output_file, index=False)

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
        if model_metrics_path.exists() \
        and 'julia' not in str(results_folder) \
        and 'neuralhydrology' not in str(results_folder) \
        and 'm0' not in str(results_folder).lower():
            # Delete the folder and merge the metrics again
            print(f"Folder 'model_metrics' already exists in {results_folder}. Deleting the folder...")
            os.system(f'rm -rf {model_metrics_path}')

        if 'julia' not in str(results_folder) \
            and 'neuralhydrology' not in str(results_folder) \
            and 'm0' not in str(results_folder).lower():

            # Merge the model metrics for the specified periods
            merge_model_metrics(results_folder, model_metrics_path, periods)

            # # Create a folder named 'model_metrics'
            # os.makedirs(model_metrics_path, exist_ok=True)

            # # There might be a bunch of single folders that have to be merged
            # # Get list of folders
            # folders = sorted([f.name for f in results_folder.iterdir() if f.is_dir()])

            # # Iterate over each period and folder
            # for period in periods:
            #     df_period = pd.DataFrame()
            #     for folder in folders:
            #         # Ensure that folder is a Path object
            #         folder_path = results_folder / folder
            #         # Construct the path for the model_metrics directory within each folder
            #         metrics_folder = folder_path / 'model_metrics'
            #         # Find a file that contains 'metrics_period' in its name within the folder / model_metrics
            #         for file in metrics_folder.glob(f'*metrics_{period}*'):
            #             if file.is_file():
            #                 # Load the file
            #                 df = pd.read_csv(file)
            #                 # Add to df_period
            #                 df_period = pd.concat([df_period, df], ignore_index=True)
                
            #     # Save the merged dataframe
            #     df_period.to_csv(model_metrics_path / f'metrics_{period}.csv', index=False)

        return model_metrics_path, results_folder

    else:
        raise FileNotFoundError(f"Folder not found: {results_folder}")

def annotate_statistics(ax, data, statistic='mean', color='tab:red', gap=0.05, fontsize=12, 
                    add_text=True, **kwargs):
    '''
    Annotate the mean or median value on the histogram plot.
    '''
    
    # data_aux = data[data > 0]
    data_aux = data.copy()
    
    if statistic == 'mean':
        value = np.mean(data_aux)
        label = f'Mean: {value:.3f}'
    elif statistic == 'median':
        value = np.median(data_aux)
        label = f'Median: {value:.3f}'
    else:
        raise ValueError("Invalid statistic. Choose 'mean' or 'median'.")

    ax.axvline(value, color=color, linestyle='--', linewidth=2)
    
    if add_text:
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
    # else:
    #     # Add the label to the legend - top right corner
    #     ax.text(0.98, 0, label, va='top', ha='right', color=color, fontsize=8)

def plot_histograms_period(metrics_path, periods, metrics, threshold_dict, 
                          graph_title, epochs, plots_folder, cluster_files=None,
                          metric_base_fname='metrics'):

    for period in periods:

        metric_file_path = metrics_path / f'{metric_base_fname}_{period}.csv'

        if metric_base_fname == 'metrics':
            # Check if the file exists and is not empty
            if metric_file_path.exists() and metric_file_path.stat().st_size <= 1:
                metric_file_path.unlink()  # Deletes the file
                print(f"Deleted empty file: {metric_file_path}")
                break

        df_period = pd.read_csv(metric_file_path)

        for metric in metrics:
            plot_metric_histogram(df_period, metric, threshold_dict, graph_title, 
                                period, epochs, plots_folder, cluster_files=cluster_files)

def plot_metric_histogram(df_period, metric, threshold_dict, graph_title, 
                          period, epochs, plots_folder, cluster_files=None):
    '''
    Plot the histogram of the given metric for the given period.
    If cluster_files is provided, plot the histograms for each cluster in subplots.
    '''
    
    # Identify the basin column - column with 'basin' in its name
    basin_column = [col for col in df_period.columns if 'basin' in col.lower()]
    # Rename the column to 'basin' if it is not already named 'basin'
    if len(basin_column) == 1:
        df_period.rename(columns={basin_column[0]: 'basin'}, inplace=True)
    else:
        raise ValueError('Column containing basin ID not found in the dataframe (at least one column should contain "basin" in its name)')

    # Filter by columns basin_ID and metrics
    df = df_period[['basin', metric]].copy()  # Explicitly create a copy

    # Plot the main histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_histogram(ax, df, metric, threshold_dict, graph_title, period, epochs)

    # Save the main plot
    fig.savefig(plots_folder / f'{metric}_histograms_{period}.png', dpi=150, bbox_inches='tight')
    
    if cluster_files is not None:
        # Plot histograms for each cluster in subplots
        fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)

        for idx, cluster_file in enumerate(cluster_files):
            with open(cluster_file, 'r') as file:
                cluster_basins = [int(line.strip()) for line in file.readlines()]

            cluster_df = df[df['basin'].isin(cluster_basins)]
            row = idx // 3
            col = idx % 3
            _plot_histogram(axes[row, col], cluster_df, metric, threshold_dict, f'Cluster {idx + 1}', period)

        # Adjust layout and save the subplot figure
        plt.tight_layout()
        fig.savefig(plots_folder / f'{metric}_histograms_{period}_clusters.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Close the figures
    plt.close('all')

def apply_threshold(df, metric, threshold_dict):
    '''
    Apply the threshold to the given metric.
    '''

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

    # If metric is NSE, replace values equal to 1.0 with 0.0
    if metric == 'nse':
        df.loc[:, metric] = df[metric].replace(1.0, 0.0)

    return df, n_below_threshold, th_value

def _plot_histogram(ax, df, metric, threshold_dict, graph_title, period, epochs=None):
    '''
    Plot the histogram of the given metric for the given period.
    '''

    if df.empty:
        print(f"Warning: DataFrame is empty for metric {metric}")
        return  # Skip plotting for this cluster if the DataFrame is empty

    if metric in threshold_dict:
        # Apply threshold to the metric
        df, n_below_threshold, th_value = apply_threshold(df, metric, threshold_dict)

    # Calculate the bins for the histogram
    hist_values = df[metric].values
    n_bins = 20
    min_value = hist_values.min()
    max_value = hist_values.max()
    bins = np.linspace(min_value, max_value, n_bins + 1)

    # Plot histogram
    ax.hist(hist_values, bins=bins, color='tab:blue', alpha=0.5)

    # Add mean and median value plot to the histogram
    if 'cluster' in graph_title.lower():
        add_text = False
    else:
        add_text = True
    annotate_statistics(ax, hist_values, statistic='mean', color='tab:red', 
                        gap=0.01, fontsize=12, add_text=add_text)
    annotate_statistics(ax, hist_values, statistic='median', color='green', 
                        gap=0.01, fontsize=12, add_text=add_text)

    # Add title, xlabel and ylabel
    n_basins = df['basin'].nunique()
    if 'cluster' in graph_title.lower():
        ax.set_title(f'{graph_title} | {n_basins} basins | {period}')
    elif metric in threshold_dict:
        if epochs is None:
            ax.set_title(f'{graph_title} (${metric.upper()} \\leq {th_value}$: ' +
                         f'{n_below_threshold}/{n_basins} basins) | {period}')
        else:
            ax.set_title(f'{graph_title} (${metric.upper()} \\leq {th_value}$: ' +
                         f'{n_below_threshold}/{n_basins} basins) | {period} | {epochs} ep')
    else:
        if epochs is None:
            ax.set_title(f'{graph_title} | {n_basins} basins | {period} | {epochs} ep')
        else:
            ax.set_title(f'{graph_title} | {n_basins} basins | {period}')

    ax.set_xlabel(f'${metric.upper()}$')
    ax.set_ylabel('Frequency')

def get_reprojected_coords(hm_catchment_path, map_shape_path, 
                           original_crs=ORIGINAL_CRS, target_crs=TARGET_CRS):
    '''
    Reproject the coordinates from the original CRS to the target CRS.
    '''

    # Load basin coordinates (hm_catchment)
    hm_catchment_gdf = gpd.read_file(hm_catchment_path)

    # Filter hm_catchment_gdf to only include the HRU_IDs in the hm_catchment['basin'] column
    # hm_catchment_gdf = hm_catchment_gdf[hm_catchment_gdf['hru_id'].isin(result_df['basin'])]
    
    # Coordinates set up
    states = gpd.read_file(map_shape_path)
    # Set the CRS for the GeoDataFrame
    states.crs = original_crs

    # Reproject the GeoDataFrame to the target CRS
    states = states.to_crs(target_crs)
    hm_catchment_gdf = hm_catchment_gdf.to_crs(target_crs)

    return states, hm_catchment_gdf

def plot_metric_map_period(metrics_path, periods, metrics, threshold_dict,
                           graph_title, epochs, plots_folder, hm_catchment_path, map_shape_path,
                           original_crs=ORIGINAL_CRS, target_crs=TARGET_CRS):
      
      states, hm_catchment_gdf = get_reprojected_coords(hm_catchment_path, map_shape_path)
    
      for period in periods:

        metric_file_path = metrics_path / f'metrics_{period}.csv'

        # Check if the file exists and is not empty
        if metric_file_path.exists() and metric_file_path.stat().st_size <= 1:
            metric_file_path.unlink()  # Deletes the file
            print(f"Deleted empty file: {metric_file_path}")
            break

        df_period = pd.read_csv(metric_file_path)

        # Column containing basin to 'basin'
        basin_column = [col for col in df_period.columns if 'basin' in col.lower()]
        if len(basin_column) == 1:
            df_period.rename(columns={basin_column[0]: 'basin'}, inplace=True)
        else:
            raise ValueError('Column containing basin ID not found in the dataframe (at least one column should contain "basin" in its name)')
    
        # Filter hm_catchment_gdf to only include the HRU_IDs in the hm_catchment['basin'] column
        hm_catchment_gdf = hm_catchment_gdf[hm_catchment_gdf['hru_id'].isin(df_period['basin'])]

        # Reproject the coordinates of the circles
        basin_coords = gpd.points_from_xy(hm_catchment_gdf['lon_cen'], hm_catchment_gdf['lat_cen'], crs=original_crs)
        basin_coords = gpd.GeoDataFrame(geometry=basin_coords, crs=original_crs)
        basin_coords = basin_coords.to_crs(target_crs)

        for metric in metrics:  
            _plot_metric_map(df_period, states, hm_catchment_gdf, 
                            basin_coords, metric, period,
                            graph_title=graph_title,
                            plots_path=plots_folder,
                            threshold_dict=threshold_dict)

def _plot_metric_map(df, states, hm_catchment_gdf, 
                    basin_coords, metric='nse', period='test',
                    graph_title='Hybrid model',
                    plots_path=Path('.'),
                    threshold_dict=None):

    # Extract data given the metric
    df = df[['basin', metric]].copy()

    # Get the threshold for the given metric, default to None
    threshold_info = threshold_dict.get(metric, None)

    if threshold_info is not None:
        th_value, th_type = threshold_info  # Unpack the threshold value and comparison type
    else:
        th_value = None

    # Define the colormap based on the metric values
    cmap = cm.oslo # choose any colormap from cmcrameri
    if threshold_info is not None:
        normalize = mcolors.Normalize(th_value, vmax=df[metric].max())
    else:
        normalize = mcolors.Normalize(vmin=df[metric].min(), vmax=df[metric].max())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot USA map
    # Define the color options
    color_options = {
        "Pale Yellow": "#FFFFE0",
        "Light Beige": "#F5F5DC",
        "Ivory": "#FFFFF0",
        "Light Yellow": "#FFFACD",
        "Lemon Chiffon": "#FFFACD",
        "Seashell": "#FFF5EE"
    }
    states.plot(ax=ax, facecolor=color_options['Pale Yellow'], edgecolor='0.2', linewidth=0.7)

    # Plot the catchments with colors based on the metric values
    hm_catchment_gdf.boundary.plot(ax=ax, color=None, alpha=0, edgecolor='gray', linewidth=0)

    if threshold_info is not None:
        # Apply threshold to the metric
        df, n_below_threshold, th_value = apply_threshold(df, metric, threshold_dict)
        ax.set_title(f'{graph_title} (${metric.upper()} \\leq {th_value}$: ' +
                                f'{n_below_threshold}/{len(df)} basins) | {period}', fontsize=16)
    else:
        ax.set_title(f'{graph_title} | {len(df)} basins) | {period}', fontsize=16)

    # Plot circles located at reprojected basin coordinates with colors based on the metric values
    ax.scatter(basin_coords.geometry.x, basin_coords.geometry.y,
            s=42,  # size of the circles
            c=df[metric],  # color based on metric values
            cmap=cmap,  # colormap
            linewidth=0.1,  # width of circle edge
            edgecolor='k',  # edge color
            alpha=1.0)  # transparency
    
    # Remove x and y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Get rid of box lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Side histogram
    # Create an axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=-0.3)

    # Set the ticks on the colorbar
    if th_value is not None:
        cbar_ticks = np.linspace(th_value, df[metric].max(), num=5)
    else:
        cbar_ticks = np.linspace(df[metric].min(), df[metric].max(), num=5)
    cbar_ticks = np.round(cbar_ticks, 2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=normalize)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax, ticks=cbar_ticks)

    # Add a histogram plot to the colorbar
    hist_values = df[metric].values
    hist_ax = divider.append_axes("right", size="10%", pad=0.6)  # adjust pad as needed

    n_bins = 20
    bins = np.linspace(hist_values.min(), hist_values.max(), n_bins + 1)

    hist_ax.hist(hist_values, bins=bins, orientation='horizontal', color='tab:blue', alpha=0.5)

    hist_ax.set_yticks([])  # remove y-ticks
    hist_ax.set_ylim(hist_values.min(), hist_values.max())  # set y-axis limits to match the data range
    # Adjust the x-ticks
    xticks = hist_ax.get_xticks()
    hist_ax.set_xticks(xticks[::3])  # set the x-ticks to every 3rd value
    hist_ax.set_xlim(0, hist_ax.get_xlim()[1])  # set the right limit to the maximum value
    # Move the x label to the top and adjust ticks
    hist_ax.set_xlabel('')
    hist_ax.xaxis.set_label_position('top')
    hist_ax.set_xlabel('Frequency', labelpad=10)
    hist_ax.xaxis.tick_top()
    hist_ax.tick_params(axis='x', which='both', labeltop=True, labelbottom=False)
    # Get rid of box lines
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['bottom'].set_visible(False)
    hist_ax.spines['left'].set_visible(False)

    # Calculate the gap height based on the current ylim
    ylim = hist_ax.get_ylim()
    gap = 0.04  # Gap between the line and the text label - histogram
    gap_height = gap * (ylim[1] - ylim[0])

    # Add mean and median value plots to the histogram
    mean_value = df[metric].mean()
    median_value = df[metric].median()

    max_value = max(mean_value, median_value)  # Find the greatest value

    # Plot mean value
    if mean_value < max_value:
        hist_ax.axhline(mean_value, color='tab:red', linestyle='--', linewidth=2)
        hist_ax.text(hist_ax.get_xlim()[1], mean_value - gap_height, f'Mean: {mean_value:.3f}', va='bottom', ha='right', color='tab:red')
    else:
        hist_ax.axhline(mean_value, color='tab:red', linestyle='--', linewidth=2)
        hist_ax.text(hist_ax.get_xlim()[1], mean_value + gap_height, f'Mean: {mean_value:.3f}', va='top', ha='right', color='tab:red')

    # Plot median value
    if median_value < max_value:
        hist_ax.axhline(median_value, color='green', linestyle='--', linewidth=2)
        hist_ax.text(hist_ax.get_xlim()[1], median_value - gap_height, f'Median: {median_value:.3f}', va='bottom', ha='right', color='green')
    else:
        hist_ax.axhline(median_value, color='green', linestyle='--', linewidth=2)
        hist_ax.text(hist_ax.get_xlim()[1], median_value + gap_height, f'Median: {median_value:.3f}', va='top', ha='right', color='green')

    # Adjust layout and save the subplot figure
    plt.tight_layout()
    fig.savefig(plots_path / f'{metric}_map_{period}.png', dpi=150, bbox_inches='tight')
    plt.show()

def load_nse_period(folder_dir, period='valid'):
    '''
    Load the metrics for the given period from the folder directory.
    '''

    model_metrics_path = folder_dir / 'model_metrics'
    metric_file = model_metrics_path / f'metrics_{period}.csv'

    if os.path.exists(metric_file):
        df = pd.read_csv(metric_file)
    else:
        csv_files = [f for f in os.listdir(folder_dir) if f.endswith('.csv')]
        df = pd.read_csv(folder_dir / csv_files[0])

    # Rename NSE column to lowercase 'nse' if it exists
    df.rename(columns={'NSE': 'nse'}, inplace=True, errors='ignore')

    # Rename basin column to 'basin'
    basin_column = [col for col in df.columns if 'basin' in col.lower()]
    if len(basin_column) == 1:
        df.rename(columns={basin_column[0]: 'basin'}, inplace=True)
    else:
        raise ValueError('Column containing basin ID not found.')

    # Make negative NSE values zero
    df['nse'] = df['nse'].apply(lambda x: max(0, x))

    return df[['basin', 'nse']].copy()

def _plot_cdf(ax, df, folder, zoom_ranges=None, markevery=20, ms=7):
    """Plot the CDF of NSE values with a horizontal line at y=0.5."""
    nse_values = np.sort(df['nse'])
    cdf = np.arange(1, len(nse_values) + 1) / len(nse_values)

    # # Print mean and median values
    # mean_value = np.mean(nse_values)
    # median_value = np.median(nse_values)
    # print(f"Median {folder['experiment']}: {median_value:.3f}")
    # print(f"Mean {folder['experiment']}: {mean_value:.3f}")
    
    # Plot the CDF line
    ax.plot(nse_values, cdf, color=folder['color'], alpha=0.8,
            linewidth=2.5, marker=folder['marker'], markevery=markevery,
            ms=ms, linestyle=folder['linestyle'], label=folder['experiment'])
    
    # Plot a solid horizontal line at y=0.5 (median line), without showing in legend
    ax.axhline(y=0.5, color='gray', linestyle='-', linewidth=0.2, label='_nolegend_')

    # Print mean and median values
    mean_value = np.mean(nse_values)
    median_value = np.median(nse_values)

    # Plot zoomed-in sections if provided
    if zoom_ranges:
        for ax_inset, zoom_range in zip(zoom_ranges['axes'], zoom_ranges['x']):
            nse_zoom = nse_values[(nse_values >= zoom_range[0]) & (nse_values <= zoom_range[1])]
            cdf_zoom = cdf[(nse_values >= zoom_range[0]) & (nse_values <= zoom_range[1])]
            ax_inset.plot(nse_zoom, cdf_zoom, color=folder['color'],
                          alpha=0.8, linewidth=2.5, marker=folder['marker'],
                          markevery=2, ms=5, linestyle=folder['linestyle'])

def plot_nse_cdf(folder4cdf_dir_list, zoom_ranges_x=None, zoom_ranges_y=None):
    # Step 1: Create and save the main plot
    fig, ax_main = plt.subplots(figsize=(10, 8))

    # Add an empty plot for the title in the legend
    ax_main.plot([], [], ' ', label=r'$\it{Experiment:}$')

    # Main plot loop
    for folder in folder4cdf_dir_list:
        folder_dir = Path(folder['directory']).expanduser()
        if folder_dir.exists():
            df = load_nse_period(folder_dir, period='valid')
            # Apply threshold to the metric NSE < 0 or NSE = -1.0 to be equal to 0
            df, _, _ = apply_threshold(df, 'nse', {'nse': [0, 'greater']})
            _plot_cdf(ax_main, df, folder)
        else:
            raise FileNotFoundError(f"Folder not found: {folder_dir}")
    
    ax_main.plot([], [], ' ', label=r'$^\dagger$ Julia/SciML')
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel('$NSE$', fontsize=14)
    ax_main.set_ylabel('CDF', fontsize=14)
    ax_main.grid(True)
    ax_main.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('cdf_nse_usa.png', dpi=150, bbox_inches='tight')
    plt.show()

    if zoom_ranges_x is None or zoom_ranges_y is None:
        return
    
    # Step 2: Create and save the zoomed-in subplot version
    fig_zoomed = plt.figure(figsize=(14, 8))
    gs = fig_zoomed.add_gridspec(3, 2, width_ratios=[2, 1])

    ax_main_zoomed = fig_zoomed.add_subplot(gs[:, 0])  # Main plot
    inset_axes = [fig_zoomed.add_subplot(gs[i, 1]) for i in range(3)]  # Zoomed-in plots

    zoom_ranges = {'axes': inset_axes, 'x': zoom_ranges_x}
    
    for folder in folder4cdf_dir_list:
        folder_dir = Path(folder['directory']).expanduser()
        if folder_dir.exists():
            df = load_nse_period(folder_dir, period='valid')
            _plot_cdf(ax_main_zoomed, df, folder, zoom_ranges)

    ax_main_zoomed.plot([], [], ' ', label=r'$^\dagger$ Julia/SciML')
    ax_main_zoomed.set_xlim(0, 1)
    ax_main_zoomed.set_ylim(0, 1)
    ax_main_zoomed.set_xlabel('$NSE$', fontsize=14)
    ax_main_zoomed.set_ylabel('CDF', fontsize=14)
    ax_main_zoomed.grid(True)
    ax_main_zoomed.legend(fontsize=12)

    # Customize each inset plot
    for ax_inset, zoom_range_x, zoom_range_y in zip(inset_axes, zoom_ranges_x, zoom_ranges_y):
        ax_inset.set_xlim(zoom_range_x)
        ax_inset.set_ylim(zoom_range_y)
        ax_inset.set_xticks(np.round(np.linspace(zoom_range_x[0], zoom_range_x[1], num=3), 2))
        ax_inset.set_yticks(np.round(np.linspace(zoom_range_y[0], zoom_range_y[1], num=3), 2))
        ax_inset.grid(True, which='major', linestyle='--', linewidth=0.7)
        ax_inset.minorticks_on()
        ax_inset.grid(True, which='minor', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('cdf_nse_usa_zoomed.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparative_histograms(folder4hist_dir_list, periods):
    '''
    Plot comparative histograms with reduced alpha from back to front and add median lines.
    
    :param folder4hist_dir_list: List of dictionaries with folder directories and plot settings (color, experiment).
    :param periods: List of periods (e.g., ['train', 'valid']) to process and plot.
    '''

    for period in periods:
        plt.figure(figsize=(10, 6))  # Create a single figure to overlay all histograms
        max_alpha = 0.7
        step_alpha = max_alpha / len(folder4hist_dir_list)
        
        for idx, folder in enumerate(folder4hist_dir_list):
            results_folder = Path(folder['directory']).expanduser()
            experiment = folder['experiment']
            
            model_metrics_path = results_folder / 'model_metrics'
            metric_file_path = model_metrics_path / f'metrics_{period}.csv'

            if model_metrics_path.exists() and 'julia' not in str(results_folder):
                # Delete the folder and merge the metrics again
                os.system(f'rm -rf {model_metrics_path}')

            if 'julia' not in str(results_folder):
                # Merge the model metrics for the specified periods
                merge_model_metrics(results_folder, model_metrics_path, periods)

            try:
                # Try reading the file first
                df_period = pd.read_csv(str(metric_file_path))
                
                # Check if the DataFrame is empty (even if the file exists and has some lines)
                if df_period.empty:
                    print(f"File is empty or contains no data: {metric_file_path}")
                    continue
            except pd.errors.EmptyDataError:
                # Handle the case where the file is completely empty or malformed
                print(f"File contains no data (EmptyDataError): {metric_file_path}")
                continue
            
            # Filter by the metric of interest
            basin_column = [col for col in df_period.columns if 'basin' in col.lower()]
            if len(basin_column) == 1:
                df_period.rename(columns={basin_column[0]: 'basin'}, inplace=True)
            
            df = df_period[['basin', 'nse']].copy()
            
            # Apply threshold to the metric NSE < 0 or NSE = -1.0 to be equal to 0
            df, _, _ = apply_threshold(df, 'nse', {'nse': [0, 'greater']})
            # df['nse'] = df['nse'].apply(lambda x: max(0, x))
            
            # Plot histogram
            hist_values = df['nse'].values
            n_bins = 20
            bins = np.linspace(hist_values.min(), hist_values.max(), n_bins + 1)

            tot_basins = len(df['nse'])
            
            alpha_value = 1.0 - (max_alpha - idx * step_alpha)
            # alpha_value = max_alpha - idx * step_alpha
            # alpha_value = 0.7
            plt.hist(hist_values, bins=bins, color=folder['color'], alpha=alpha_value, label=f"{experiment} | {tot_basins} basins | {period}")
            
            # Remove 'tab' from the color string if it exists
            # line_color = folder['color'].replace('tab:', '')
            line_color = folder['color']

            # Calculate and plot the median
            median_value = np.median(hist_values)
            print(f"Median {experiment} ({period}): {median_value:.3f}")
            plt.axvline(median_value, color=line_color, linestyle='--', linewidth=2)

            # Calculate and plot the mean
            mean_value = np.mean(hist_values)
            print(f"Mean {experiment} ({period}): {mean_value:.3f}")
            plt.axvline(mean_value, color=line_color, linestyle=':', linewidth=2)
        
        # ax_main.plot([], [], ' ', label=r'$^\dagger$ Julia/SciML')
        plt.plot([], [], '--', label='Median', color='black')
        plt.plot([], [], ':', label='Mean', color='black')

        plt.xlabel('$NSE$')
        plt.ylabel('Frequency')
        plt.title(f'Comparison of NSE across models for $\\mathbf{{{period}}}$ period')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save the plot
        plt.savefig(f'nse_histograms_{period}.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    pass