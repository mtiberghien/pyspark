import time
from os.path import isdir
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import DataFrame
from abc import ABC, abstractmethod


def write_training_time(path: str, training_time):
    with open("{}/training_time.txt".format(path), 'w') as f:
        f.write('{:.2f}'.format(training_time))


def read_training_time(path: str):
    with open("{}/training_time.txt".format(path)) as f:
        result = float(f.read())
    return result


def read_models_number(path: str):
    with open("{}/models_number.txt".format(path)) as f:
        result = int(f.read())
    return result


def write_models_number(path: str, models_number):
    with open("{}/models_number.txt".format(path), 'w') as f:
        f.write(str(models_number))


# Abstract model builder class. It loads or train a model using cross validation
class ModelBuilder(ABC):
    def __init__(self, df_train: DataFrame, model_name: str):
        self.model_name = model_name
        path = 'modeles/{}'.format(self.model_name)
        self.model = None
        self.training_time = 0
        self.models_number = 0
        # If the model exists it is loaded using abstract method 'load_model'
        if isdir(path):
            self.model = self.load_model(path)
            self.training_time = read_training_time(path)
            self.models_number = read_models_number(path)
        # If the model doesn't exist is is instantiated using abstract method 'instantiate_model'
        else:
            model = self.instantiate_model()
            # The ParamGrid for cross validation is built using abstract method 'build_param_grid'
            param_grid = self.build_param_grid(model)
            cross_validation = CrossValidator(estimator=model,
                                              estimatorParamMaps=param_grid,
                                              evaluator=BinaryClassificationEvaluator(),
                                              numFolds=3)
            self.models_number = len(param_grid)
            start = time.time()
            self.model = cross_validation.fit(df_train).bestModel
            # Training time is measured for analysis
            self.training_time = time.time() - start
            self.model.save(path)
            write_training_time(path, self.training_time)
            write_models_number(path, self.models_number)
        show_best_model(self)

    # load a persisted model
    @abstractmethod
    def load_model(self, path: str):
        pass

    # instantiate a binary classifier model
    @abstractmethod
    def instantiate_model(self):
        pass

    # instantiate the ParamGrid used by cross validation to select the best parameters
    @abstractmethod
    def build_param_grid(self, parent_model):
        pass

    def show_best_params(self):
        print("Best parameters:")


def show_best_model(model_builder: ModelBuilder):
    print('-------------------------------------')
    print("Best {} among {} in {} seconds".format(model_builder.model_name, model_builder.models_number,
                                                  model_builder.training_time))
    model_builder.show_best_params()
    print('-------------------------------------')