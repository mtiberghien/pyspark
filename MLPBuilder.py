from pyspark.ml.classification import MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from utils import read_training_time, write_training_time, read_models_number, write_models_number, show_best_model
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from os.path import isdir
import time


class MLPBuilder:
    def __init__(self, training_df: DataFrame, n_features, model_name='MLP'):
        self.model_name = model_name
        if isdir(self.model_name):
            self.model = MultilayerPerceptronClassificationModel.load(self.model_name)
            self.training_time = read_training_time(self.model_name)
            self.models_number = read_models_number(self.model_name)
        else:
            mlp = MultilayerPerceptronClassifier(maxIter=100, stepSize=0.1, layers=[30, 2])
            param_grid = ParamGridBuilder() \
                .addGrid(mlp.stepSize, [0.001, 0.01, 0.1]) \
                .addGrid(mlp.layers, [[n_features, 2], [n_features, n_features*2, 2],
                                      [n_features, n_features*2, n_features*2, 2],
                                      [n_features, n_features*2, n_features*4, n_features, 2]]) \
                .build()
            cross_validation = CrossValidator(estimator=mlp,
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
