from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from utils import assemble_features, scale_features


class Preprocessor:
    def __init__(self, dataframe: DataFrame, true_condition, label_column: str, features_columns, normalize=True):
        self.dataframe = assemble_features(dataframe, features_columns)
        if normalize:
            self.dataframe = scale_features(self.dataframe, True, True)
        self.dataframe = self.dataframe.withColumn("label", true_condition(col(label_column)))
        features = col('scaled_features').alias('features') if normalize else col('features')
        self.dataframe = self.dataframe.select("label", features)
