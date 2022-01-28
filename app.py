from pyspark.sql.session import SparkSession
from Preprocessor import Preprocessor
from utils import corr_features
import seaborn as sns
import matplotlib.pyplot as plt

spark = (SparkSession.builder.getOrCreate())

df = spark.read.options(header=True, inferSchema='True').csv("data/breast cancer.csv")

preprocessor = Preprocessor(df, lambda d: (d == 'M').cast('float'), 'diagnosis', df.columns[2:], True)

(df_train, df_test) = preprocessor.dataframe.randomSplit([0.8, 0.2])

correlation_matrix = corr_features(preprocessor.dataframe)
sns.heatmap(correlation_matrix)
plt.savefig('correlation_matrix.png')
plt.show()

