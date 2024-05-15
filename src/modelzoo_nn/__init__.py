from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel
from src.modelzoo_nn.mlp import MLP
from src.modelzoo_nn.pretrainer import NNpretrainer


def get_nn_model(concept_model: BaseConceptModel, alias_map:dict)-> BaseNNModel:
    '''Get the neural network model based on the configuration'''
    
    if concept_model.cfg.nn_model.lower() == "mlp":
        Model = MLP
    else:
        raise NotImplementedError(f"No model NN class implemented for model {concept_model.cfg.nn_model}")
    
    return Model(concept_model=concept_model, alias_map=alias_map)

def get_nn_pretrainer(nnmodel: BaseNNModel, input_vars=None, output_vars=None):
    '''Get the pretrainer for the neural network model'''
    
    return NNpretrainer(nnmodel, input_vars=input_vars, output_vars=output_vars)
                 