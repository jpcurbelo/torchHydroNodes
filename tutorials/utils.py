import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_latest_run_folder(runs_dir="runs"):
    """
    Finds the most recently created folder in the specified directory based on the YYMMDD_HHMMSS timestamp.

    Args:
        runs_dir (str): Path to the directory containing run folders.

    Returns:
        str: Path to the most recently created folder.
    """
    runs_path = Path(runs_dir)
    
    # Check if the directory exists
    if not runs_path.exists() or not runs_path.is_dir():
        raise ValueError(f"The directory '{runs_dir}' does not exist or is not a valid directory.")
    
    # Get all directories in the 'runs' directory with a timestamp
    folders = [
        folder for folder in runs_path.iterdir() 
        if folder.is_dir() and "_" in folder.name and folder.name.split("_")[-1].isdigit()
    ]
    
    # Sort directories by their timestamps (extracted from folder names)
    sorted_folders = sorted(folders, key=lambda f: f.name.split("_")[-1], reverse=True)
    
    if not sorted_folders:
        raise ValueError("No valid run folders found in the directory.")
    
    # Return the most recent folder
    return str(sorted_folders[0])

def display_run_plots(runs_dir="runs", basins=[], 
                      periods=["train", "valid"],
                      limit=None):
    """
    Displays plots from the latest run folder, creating a separate figure for each basin,
    with one row for each period.

    Args:
        runs_dir (str): Path to the directory containing run folders.
        basins (list): List of basin identifiers.
        periods (list): List of periods (e.g., ["train", "valid"]).
        limit (int, optional): Limit the number of basins to display. Defaults to None.
    """
    # Get the latest run folder
    latest_folder = get_latest_run_folder(runs_dir)
    print("Latest folder:", latest_folder)

    # Limit the number of basins if specified
    if limit is not None:
        basins = sorted(basins)[:limit]

    for basin in basins:
        # Create a new figure for each basin
        num_cols = len(periods)
        fig, axes = plt.subplots(1, num_cols, figsize=(10 * num_cols, 5))

        # Ensure axes is iterable for single column cases
        if num_cols == 1:
            axes = [axes]

        for j, period in enumerate(periods):
            # Construct the image path
            image_path = Path(latest_folder) / "model_plots" / f"{basin}_{period}.png"

            # Display the image if it exists
            ax = axes[j]
            if image_path.exists():
                img = mpimg.imread(image_path)
                ax.imshow(img)
                ax.axis('off')
            else:
                ax.axis('off')
                ax.set_title(f"missing: {period}")

        # Add a title for the entire figure
        fig.suptitle(f"Basin: {basin}", fontsize=14)  
        plt.tight_layout() 
        plt.show()

if __name__ == '__main__':
    pass