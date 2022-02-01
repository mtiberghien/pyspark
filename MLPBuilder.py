from pyspark.ml.classification import MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import DataFrame
from utils import ModelBuilder


# MLP model builder
class MLPBuilder(ModelBuilder):
    def __init__(self, df_train: DataFrame, n_features, model_name='MLP'):
        self.n_features = n_features
        super().__init__(df_train, model_name)

    def load_model(self, path: str):
        return MultilayerPerceptronClassificationModel.load(path)

    def instantiate_model(self):
        return MultilayerPerceptronClassifier(maxIter=100, stepSize=0.1, layers=[self.n_features, 2])

    def build_param_grid(self, parent_model):
        return ParamGridBuilder() \
            .addGrid(parent_model.stepSize, [0.001, 0.01, 0.1]) \
            .addGrid(parent_model.layers, [[self.n_features, 2], [self.n_features, self.n_features * 2, 2],
                                           [self.n_features, self.n_features * 2, self.n_features * 2, 2],
                                           [self.n_features, self.n_features * 2, self.n_features * 4, self.n_features,
                                            2]]) \
            .build()

    def show_best_params(self):
        super().show_best_params()
        print("\tStepSize:{}".format(self.model.getStepSize()))
        print("\tLayers:{}".format(self.model.getLayers()))
