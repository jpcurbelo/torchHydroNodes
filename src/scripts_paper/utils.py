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
    basin_repeated = []
    basin_folder_list = os.listdir(runs_path)
    print(f'Checking finished basins in {runs_folder}...', len(basin_folder_list))
    # Remove model_metrics folder from basin_folder_list if it is present
    if 'model_metrics' in basin_folder_list:
        basin_folder_list.remove('model_metrics')

    for folder in basin_folder_list:

        basin = get_basin_id(folder)

        if job_is_finished(runs_path / folder):
            if basin in basin_finished:
                basin_repeated.append(basin)
            basin_finished.append(str(int(basin)))
        else:
            # if basin is not None:
            basin_unfinished.append(basin)

    print('Repeated basins:', basin_repeated)
    print('Finished basins:', len(basin_finished))
    print('Unfinished basins:', len(basin_unfinished))

    return sorted(basin_finished), sorted(basin_unfinished)

def delete_unfinished_jobs(runs_folder, basins):

    # Find folders that contain basin in their name
    folders = [f for f in os.listdir(runs_folder) if any(basin in f for basin in basins)]
    
    # Delete the folders
    for folder in folders:
        os.system(f'rm -rf {runs_folder / folder}')


if __name__ == "__main__":
    pass