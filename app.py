from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import PCA, PCAModel
from MLPBuilder import MLPBuilder
from PCAFeatureSelector import PCAFeatureSelector
from Preprocessor import Preprocessor
from RandomForestBuilder import RandomForestBuilder
from SVMBuilder import SVMBuilder
from utils import corr_features, evaluate_model
import seaborn as sns
import matplotlib.pyplot as plt
from LogisticRegressionBuilder import LogisticRegressionBuilder
import numpy as np

spark = (SparkSession.builder.getOrCreate())


def preprocess_data(dataframe: DataFrame):
    preprocessor = Preprocessor(dataframe, lambda d: (d == 'M').cast('float'), 'diagnosis', dataframe.columns[2:], True)
    return preprocessor.dataframe


def save_correlation_matrix(dataframe: DataFrame):
    correlation_matrix = corr_features(dataframe)
    sns.heatmap(correlation_matrix)
    plt.savefig('correlation_matrix.png')
    plt.show()


breast_cancer_data = spark.read.options(header=True, inferSchema='True').csv("data/breast cancer.csv")
n_features = len(breast_cancer_data.columns[2:])

df = preprocess_data(breast_cancer_data)

# done once for analysis
# save_correlation_matrix(df)

(df_train, df_test) = df.randomSplit([0.8, 0.2])

# Logistic Regression training and evaluation
lrb = LogisticRegressionBuilder(df_train)
evaluate_model(df_test, lrb)

# Random Forest training and evaluation
rfb = RandomForestBuilder(df_train)
evaluate_model(df_test, rfb)

# SVM training and evaluation
svm_builder = SVMBuilder(df_train)
evaluate_model(df_test, svm_builder)

# MLP training and evaluation
mlp_builder = MLPBuilder(df_train, n_features)
evaluate_model(df_test, mlp_builder)

# Test results using PCA feature reduction
pca = PCAFeatureSelector(df, n_features)

(df_train_pca, df_test_pca) = pca.df_pca.randomSplit([0.8, 0.2])

# Logistic Regression PCA training and evaluation
lrb_pca = LogisticRegressionBuilder(df_train_pca, 'Logistic Regression PCA')
evaluate_model(df_test_pca, lrb_pca)

# Random Forest PCA training and evaluation
rfb_pca = RandomForestBuilder(df_train_pca, 'Random Forest PCA')
evaluate_model(df_test_pca, rfb_pca)

# SVM training and evaluation
svm_builder_pca = SVMBuilder(df_train_pca, 'Linear SVC PCA')
evaluate_model(df_test_pca, svm_builder_pca)

# MLP training and evaluation
mlp_builder_pca = MLPBuilder(df_train_pca, pca.n_features, model_name='MLP PCA')
evaluate_model(df_test_pca, mlp_builder_pca)



