import numpy as np
from pyspark.ml.feature import PCAModel, PCA
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from utils import scale_features


class PCAFeatureSelector:
    def __init__(self, df: DataFrame, n_features_init, min_variance=0.95):
        pca_model: PCAModel = PCA(k=n_features_init, inputCol='features').fit(df)
        for i in range(n_features_init):
            self.explained_variance = np.sum(pca_model.explainedVariance[:i + 1])
            if self.explained_variance >= min_variance:
                break
        self.n_features = i + 1
        print("Projected {} features over {} using PCA to explain {:.2f}% of variance".format(self.n_features,
                                                                                              n_features_init,
                                                                                              self.explained_variance * 100))
        self.df_pca = PCA(k=self.n_features, inputCol='features', outputCol='pca_features') \
            .fit(df).transform(df).select('label', col('pca_features').alias('features'))
        self.df_pca = scale_features(self.df_pca, True, True).select('label', col('scaled_features').alias('features'))
