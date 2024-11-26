from src.utils.load_process_data import Config
from src.modelzoo_nn.basepretrainer import NNpretrainer
from src.datasetzoo.basedataset import BaseDataset
from src.modelzoo_hybrid.exphydroM100 import (
    BaseHybridModel, 
    ExpHydroM100,
)
from src.modelzoo_hybrid.basetrainer import BaseHybridModelTrainer

def get_hybrid_model(cfg: Config, pretrainer: NNpretrainer, dataset: BaseDataset):
    '''Get the hybrid model'''
    
    if cfg.hybrid_model.lower() == "exphydrom100":
        Model = ExpHydroM100
    else:
        raise NotImplementedError(f"No hybrid model class implemented for model {cfg.hybrid_model.lower()}")

    model_instance = Model(cfg=cfg, pretrainer=pretrainer, ds=dataset.ds_train, scaler=dataset.scaler)
    return model_instance

def get_trainer(model: BaseHybridModel):
    '''Get the trainer for the hybrid model'''
    return BaseHybridModelTrainer(model=model)