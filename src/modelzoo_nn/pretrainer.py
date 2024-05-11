

class NNpretrainer:

    def __init__(self, nnmodel, input_vars=None, output_vars=None) -> None:
        
        self.nnmodel = nnmodel

        # print(pretrainer.nnmodel.concept_model.ds)
        # print(pretrainer.nnmodel.concept_model.ds.basin.values)
        self.dataset = self.nnmodel.concept_model.ds
        self.basins = self.dataset.basin.values
        self.input_vars = input_vars
        self.output_vars = output_vars

        # 
