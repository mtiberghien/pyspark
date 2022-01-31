from pyspark.ml.evaluation import BinaryClassificationEvaluator
from utils import read_training_time, write_training_time, read_models_number, write_models_number, show_best_model
from pyspark.ml.classification import LinearSVC, LinearSVCModel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from os.path import isdir
import time


class SVMBuilder:
    def __init__(self, training_df: DataFrame, model_name='Linear SVC'):
        self.model_name = model_name
        if isdir(self.model_name):
            self.model = LinearSVCModel.load(self.model_name)
            self.training_time = read_training_time(self.model_name)
            self.models_number = read_models_number(self.model_name)
        else:
            svc = LinearSVC(maxIter=100, regParam=0.1)
            param_grid = ParamGridBuilder() \
                .addGrid(svc.regParam, [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]) \
                .build()
            cross_validation = CrossValidator(estimator=svc,
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
