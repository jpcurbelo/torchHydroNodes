from src.datasetzoo.basedataset import BaseDataset
from src.datasetzoo.camelsus import CamelsUS

def get_dataset(cfg: dict,
                is_train: bool,
                period: str,
                basin: str = None,
                scaler: dict = {}) -> BaseDataset:
    
    if cfg['dataset'].lower() == "camelsus":
        Dataset = CamelsUS
        
    else:
        raise NotImplementedError(f"No dataset class implemented for dataset {cfg.dataset}")
    
    ds = Dataset(cfg=cfg, 
                 is_train=is_train, 
                 period=period, 
                 basin=basin, 
                 scaler=scaler)
    
    return ds
