from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import DataFrame
from pyspark.ml.stat import Correlation
import seaborn as sns
import matplotlib.pyplot as plt


def assemble_features(dataframe: DataFrame, features_columns):
    assembler = VectorAssembler(inputCols=features_columns, outputCol='features')
    return assembler.transform(dataframe)


def scale_features(dataframe: DataFrame, with_mean=True, with_std=True):
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=with_mean, withStd=with_std).fit(
        dataframe)
    return scaler.transform(dataframe)


def corr_features(dataframe: DataFrame, method='pearson'):
    return Correlation.corr(dataframe, "features", method=method).head()[0].toArray()


def write_training_time(path:str, training_time):
    with open("{}/training_time.txt".format(path), 'w') as f:
        f.write('{:.2f}'.format(training_time))


def read_training_time(path:str):
    with open("{}/training_time.txt".format(path)) as f:
        result = float(f.read())
    return result


def read_models_number(path:str):
    with open("{}/models_number.txt".format(path)) as f:
        result = int(f.read())
    return result


def write_models_number(path:str, models_number):
    with open("{}/models_number.txt".format(path), 'w') as f:
        f.write(str(models_number))


def show_best_model(model):
    print("Got best {} among {} in {} seconds".format(model.model_name, model.models_number, model.training_time))


def show_confusion_matrix(predictions: DataFrame, model_name: str):
    metrics = MulticlassMetrics(predictions.select('prediction', 'label').rdd)
    print('Accuracy for {}: {:.2f}%'.format(model_name, metrics.accuracy*100))
    sns.heatmap(metrics.confusionMatrix().toArray(), annot=True)
    plt.savefig('{}.png'.format(model_name))
    plt.show()


def show_auc(predictions: DataFrame, model_name: str):
    print("AUC for {}: {}".format(model_name, BinaryClassificationEvaluator().evaluate(predictions)))


def evaluate_model(df_test: DataFrame, model_builder):
    lr_predictions = model_builder.model.transform(df_test)
    show_auc(lr_predictions, model_builder.model_name)
    show_confusion_matrix(lr_predictions, model_builder.model_name)

