import xarray

from src.utils.load_process import Config
from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_concept.exphydro import ExpHydro

def get_concept_model(cfg: Config,
                      ds: xarray.Dataset,
                      scaler: None,
                      odesmethod:str='RK23'    # 'RK45'     #'RK23'
                    ) -> BaseConceptModel:
    '''Get the concept model based on the configuration'''
    
    if cfg.concept_model.lower() == "exphydro":
        Model = ExpHydro
    else:
        raise NotImplementedError(f"No conceptual model class implemented for model {cfg.concept_model}")
    
    return Model(cfg=cfg, ds=ds, scaler=scaler, odesmethod=odesmethod)