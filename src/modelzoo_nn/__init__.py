from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel
from src.modelzoo_nn.mlp import MLP
from src.modelzoo_nn.lstm import LSTM
from src.modelzoo_nn.basepretrainer import NNpretrainer
from src.datasetzoo.basedataset import BaseDataset
import xarray


def get_nn_model(concept_model: BaseConceptModel, ds_static: xarray.Dataset)-> BaseNNModel:
    '''Get the neural network model based on the configuration'''

    # print('concept_model.cfg.nn_model.lower()', concept_model.cfg.nn_model.lower())
    
    if concept_model.cfg.nn_model.lower() == "mlp":
        Model = MLP
    elif concept_model.cfg.nn_model.lower() == "lstm":
        Model = LSTM
    else:
        raise NotImplementedError(f"No model NN class implemented for model {concept_model.cfg.nn_model}")
    
    return Model(concept_model=concept_model, ds_static=ds_static)

def get_nn_pretrainer(nnmodel: BaseNNModel, fulldataset: BaseDataset):
    '''Get the pretrainer for the neural network model'''
    
    return NNpretrainer(nnmodel=nnmodel, fulldataset=fulldataset)
                 