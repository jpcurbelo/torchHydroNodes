from pathlib import Path

from src.utils.load_process_data import Config
from src.datasetzoo.basedataset import BaseDataset
from src.datasetzoo.camelsus import CamelsUS
from src.datasetzoo.pretrainer import Pretrainer

# Get the absolute path to the current script
script_dir = Path(__file__).resolve().parent

def get_dataset(cfg: Config,
                is_train: bool,
                scaler: dict = dict()) -> BaseDataset:
    '''
    Function to get the dataset object for Conceptual models - M0
    
    - Args:
        - cfg: Config object with the configuration parameters
        - is_train: Boolean indicating if the dataset is for training or testing
        - scaler: Dictionary with the scaler object
    
    - Returns:
        - ds: BaseDataset object
    '''
    
    if cfg.dataset.lower() == "camelsus":
        Dataset = CamelsUS
        
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")
    
    ds = Dataset(cfg=cfg, is_train=is_train, scaler=scaler)
    
    return ds

def get_dataset_pretrainer(cfg: Config,
                           scaler: dict = dict()) -> BaseDataset:
    '''
    Function to get the dataset object for the pretrainer
    
    - Args:
        - cfg: Config object with the configuration parameters
        - scaler: Dictionary with the scaler object
    
    - Returns:
        - ds: BaseDataset object
    '''

    print('script_dir', script_dir) 
    print('script_dir.parent.parent', script_dir.parent.parent)

    data_path = script_dir.parent.parent / 'data' / cfg.data_dir
    
    # Check if datadir exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} not found")

    cfg.data_dir = data_path
    return Pretrainer(cfg=cfg, scaler=scaler)
