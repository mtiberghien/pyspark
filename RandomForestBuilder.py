from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import DataFrame
from utils import ModelBuilder


# Random Forest model builder
class RandomForestBuilder(ModelBuilder):
    def __init__(self, df_train: DataFrame, model_name='Random Forest'):
        super().__init__(df_train, model_name)

    def load_model(self, path: str):
        return RandomForestClassificationModel.load(path)

    def instantiate_model(self):
        return RandomForestClassifier(impurity='gini', numTrees=400)

    def build_param_grid(self, parent_model):
        return ParamGridBuilder() \
                .addGrid(parent_model.impurity, ['gini', 'entropy']) \
                .addGrid(parent_model.maxDepth, [3, 5, 10, 15, 20]) \
                .build()

    def show_best_params(self):
        super().show_best_params()
        print("\tImpurity:{}".format(self.model.getImpurity()))
        print("\tMaxDepth:{}".format(self.model.getMaxDepth()))
