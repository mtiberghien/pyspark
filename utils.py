from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import DataFrame
from pyspark.ml.stat import Correlation


def assemble_features(dataframe: DataFrame, features_columns):
    assembler = VectorAssembler(inputCols=features_columns, outputCol='features')
    return assembler.transform(dataframe)


def scale_features(dataframe: DataFrame, with_mean=True, with_std=True):
    scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withMean=with_mean, withStd=with_std).fit(
        dataframe)
    return scaler.transform(dataframe)


def corr_features(dataframe: DataFrame, method='pearson'):
    return Correlation.corr(dataframe, "features", method=method).head()[0].toArray()
