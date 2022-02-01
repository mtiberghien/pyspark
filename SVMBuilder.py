from utils import ModelBuilder
from pyspark.ml.classification import LinearSVC, LinearSVCModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame


# SVM model builder
class SVMBuilder(ModelBuilder):
    def __init__(self, df_train: DataFrame, model_name='Linear SVC'):
        super().__init__(df_train, model_name)

    def load_model(self, path: str):
        return LinearSVCModel.load(path)

    def instantiate_model(self):
        return LinearSVC(maxIter=100, regParam=0.1)

    def build_param_grid(self, parent_model):
        return ParamGridBuilder() \
                .addGrid(parent_model.regParam, [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]) \
                .build()

    def show_best_params(self):
        super().show_best_params()
        print("\tRegParam:{}".format(self.model.getRegParam()))
