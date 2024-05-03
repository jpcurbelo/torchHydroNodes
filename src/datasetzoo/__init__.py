from src.utils.load_process import Config
from src.datasetzoo.basedataset import BaseDataset
from src.datasetzoo.camelsus import CamelsUS

def get_dataset(cfg: Config,
                is_train: bool,
                scaler: dict = dict()) -> BaseDataset:
    
    if cfg.dataset.lower() == "camelsus":
        Dataset = CamelsUS
        
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")
    
    ds = Dataset(cfg=cfg, 
                 is_train=is_train, 
                 scaler=scaler)
    
    return ds
