from utils import ModelBuilder
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import DataFrame


# Logistic Regression model builder
class LogisticRegressionBuilder(ModelBuilder):
    def __init__(self, df_train: DataFrame, model_name='Logistic Regression'):
        super().__init__(df_train, model_name)

    def load_model(self, path: str):
        return LogisticRegressionModel.load(self.model_name)

    def instantiate_model(self):
        return LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)

    def build_param_grid(self, parent_model):
        return ParamGridBuilder() \
            .addGrid(parent_model.regParam, [0.1, 0.01, 0.001]) \
            .addGrid(parent_model.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]) \
            .build()

    def show_best_params(self):
        super().show_best_params()
        print("RegParam:{}".format(self.model.getRegParam()))
        print("ElasticNetParam:{}".format(self.model.getElasticNetParam()))
