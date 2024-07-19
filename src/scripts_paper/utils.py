import re
import os
from pathlib import Path

def get_basin_id(folder_name:str):
    '''
    Extract the basin name from the folder name
    
    - Args:
        - folder_name: str, folder name
        
    - Returns:
        - str, basin id
    '''
    
    # Extract the basin name from the nn_model_dir
    # Use a regular expression to find an 8-digit number in the string
    match = re.search(r'\d{8}', folder_name)
    if match:
        basin = match.group(0).strip()
    else:
        basin = None

    return basin

def job_is_finished(folder_path:str):
    '''
    Check if the job is finished
    
    - Args:
        - folder: str, folder name
        
    - Returns:
        - bool, True if the job is finished, False otherwise
    '''
    # Check if the folder contains the file 'job_finished.txt'
    metric_path = folder_path / 'model_metrics' 
    if os.path.exists(metric_path) and len(os.listdir(metric_path)) > 1:
        return True
    else:
        return False

def check_finished_basins(runs_path):

    # Extract runs folder
    runs_folder = str(runs_path).split('/')[-1]

    basin_finished = []
    basin_unfinished = []
    for folder in os.listdir(runs_folder):

        basin = get_basin_id(folder)
        if job_is_finished(runs_path / folder):
            basin_finished.append(str(int(basin)))
        else:
            basin_unfinished.append(basin)

    return sorted(basin_finished), sorted(basin_unfinished)

def delete_unfinished_jobs(runs_folder, basins):

    # Find folders that contain basin in their name
    folders = [f for f in os.listdir(runs_folder) if any(basin in f for basin in basins)]
    
    # Delete the folders
    for folder in folders:
        os.system(f'rm -rf {runs_folder / folder}')


if __name__ == "__main__":
    pass