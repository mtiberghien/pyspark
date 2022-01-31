from pyspark.ml.evaluation import BinaryClassificationEvaluator
from utils import read_training_time, write_training_time, read_models_number, write_models_number, show_best_model
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from os.path import isdir
import time


class LogisticRegressionBuilder:
    def __init__(self, training_df: DataFrame, model_name='Logistic Regression'):
        self.model_name = model_name
        if isdir(self.model_name):
            self.model = LogisticRegressionModel.load(self.model_name)
            self.training_time = read_training_time(self.model_name)
            self.models_number = read_models_number(self.model_name)
        else:
            lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)
            param_grid = ParamGridBuilder() \
                .addGrid(lr.regParam, [0.1, 0.01, 0.001]) \
                .addGrid(lr.elasticNetParam, [0, 0.25, 0.5, 0.75, 1]) \
                .build()
            cross_validation = CrossValidator(estimator=lr,
                                              estimatorParamMaps=param_grid,
                                              evaluator=BinaryClassificationEvaluator(),
                                              numFolds=3)
            self.models_number = len(param_grid)
            start = time.time()
            self.model = cross_validation.fit(training_df).bestModel
            self.training_time = time.time() - start
            self.model.save(self.model_name)
            write_training_time(self.model_name, self.training_time)
            write_models_number(self.model_name, self.models_number)
        show_best_model(self)
