import time
from os.path import isdir
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import DataFrame

from utils import read_training_time, read_models_number, write_training_time, write_models_number, show_best_model


class RandomForestBuilder:
    def __init__(self, df_train: DataFrame, model_name='Random Forest'):
        self.model_name = model_name
        if isdir(self.model_name):
            self.model = RandomForestClassificationModel.load(self.model_name)
            self.training_time = read_training_time(self.model_name)
            self.models_number = read_models_number(self.model_name)
        else:
            rf = RandomForestClassifier(impurity='gini', numTrees=400)
            param_grid = ParamGridBuilder() \
                .addGrid(rf.impurity, ['gini', 'entropy']) \
                .addGrid(rf.maxDepth, [3, 5, 10, 15, 20]) \
                .build()
            cross_validation = CrossValidator(estimator=rf,
                                              estimatorParamMaps=param_grid,
                                              evaluator=BinaryClassificationEvaluator(),
                                              numFolds=3)
            self.models_number = len(param_grid)
            start = time.time()
            self.model = cross_validation.fit(df_train).bestModel
            self.training_time = time.time() - start
            self.model.save(self.model_name)
            write_training_time(self.model_name, self.training_time)
            write_models_number(self.model_name, self.models_number)
        show_best_model(self)
