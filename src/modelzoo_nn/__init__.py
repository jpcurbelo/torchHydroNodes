from src.modelzoo_concept.basemodel import BaseConceptModel
from src.modelzoo_nn.basemodel import BaseNNModel
from src.modelzoo_nn.mlp import MLP


def get_nn_model(concept_model: BaseConceptModel)-> BaseNNModel:
    '''Get the neural network model based on the configuration'''
    
    if concept_model.cfg.nn_model.lower() == "mlp":
        Model = MLP
    else:
        raise NotImplementedError(f"No model NN class implemented for model {concept_model.cfg.nn_model}")
    
    return Model(concept_model=concept_model)
                 