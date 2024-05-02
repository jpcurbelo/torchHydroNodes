import xarray

from src.utils.utils_load_process import Config
from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_concept.exhydro import ExpHydro

def get_concept_model(cfg: Config,
                      ds: xarray.Dataset,
                      odesmethod:str ='RK23'
                    ) -> BaseConceptModel:
    '''Get the concept model based on the configuration'''
    
    if cfg.concept_model.lower() == "exphydro":
        model = ExpHydro(cfg,
                         ds,
                         odesmethod=odesmethod)        
    else:
        raise NotImplementedError(f"No model class implemented for model {cfg.model}")