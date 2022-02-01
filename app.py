from os.path import isfile
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from MLPBuilder import MLPBuilder
from PCAFeatureSelector import PCAFeatureSelector
from Preprocessor import Preprocessor
from RandomForestBuilder import RandomForestBuilder
from SVMBuilder import SVMBuilder
from utils import save_correlation_matrix, evaluate_model
from LogisticRegressionBuilder import LogisticRegressionBuilder

# Initialize spark session
spark = SparkSession.builder. \
    master('local'). \
    appName('breast_cancer'). \
    getOrCreate()

spark.sparkContext.setLogLevel('ERROR')


# Preprocessing breast cancer data
def preprocess_breast_data(dataframe: DataFrame):
    preprocessor = Preprocessor(dataframe, lambda d: (d == 'M').cast('float'), 'diagnosis', dataframe.columns[2:], True)
    return preprocessor.dataframe


# Read csv file
breast_cancer_data = spark.read.options(header=True, inferSchema='True').csv("data/breast cancer.csv")

# Keep trace of original number of features
n_features = len(breast_cancer_data.columns[2:])

# Preprocessor is creating a dataframe with label as binary classification and features as vector
df = preprocess_breast_data(breast_cancer_data)

# done once for analysis
if not isfile('images/correlation_matrix.png'):
    save_correlation_matrix(df)

(df_train, df_test) = df.randomSplit([0.8, 0.2])

# Logistic Regression training or loading and evaluation
lrb = LogisticRegressionBuilder(df_train)
evaluate_model(df_test, lrb)

# Random Forest training or loading and evaluation
rfb = RandomForestBuilder(df_train)
evaluate_model(df_test, rfb)

# SVM training or loading and evaluation
svm_builder = SVMBuilder(df_train)
evaluate_model(df_test, svm_builder)

# MLP training or loading and evaluation
mlp_builder = MLPBuilder(df_train, n_features)
evaluate_model(df_test, mlp_builder)

# Test results using PCA feature reduction
pca = PCAFeatureSelector(df, n_features)

(df_train_pca, df_test_pca) = pca.df_pca.randomSplit([0.8, 0.2])

# Logistic Regression PCA training or loading and evaluation
lrb_pca = LogisticRegressionBuilder(df_train_pca, 'Logistic Regression PCA')
evaluate_model(df_test_pca, lrb_pca)

# Random Forest PCA training or loading and evaluation
rfb_pca = RandomForestBuilder(df_train_pca, 'Random Forest PCA')
evaluate_model(df_test_pca, rfb_pca)

# SVM PCA training or loading and evaluation
svm_builder_pca = SVMBuilder(df_train_pca, 'Linear SVC PCA')
evaluate_model(df_test_pca, svm_builder_pca)

# MLP PCA training or loading and evaluation
mlp_builder_pca = MLPBuilder(df_train_pca, pca.n_features, model_name='MLP PCA')
evaluate_model(df_test_pca, mlp_builder_pca)
