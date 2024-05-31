from src.utils.load_process_data import Config
from src.modelzoo_nn.basemodel import BaseNNModel
from src.datasetzoo.basedataset import BaseDataset
from src.modelzoo_hybrid.expohydroM100 import ExpHydroM100

def get_hybrid_model(cfg: Config, pretrainer: BaseNNModel, dataset: BaseDataset):
    '''Get the hybrid model'''
    
    if cfg.hybrid_model.lower() == "exphydrom100":
        Model = ExpHydroM100
    else:
        raise NotImplementedError(f"No hybrid model class implemented for model {cfg.hybrid_model.lower()}")
    
    return Model(cfg=cfg, nnmodel=pretrainer, ds=dataset.ds_train)