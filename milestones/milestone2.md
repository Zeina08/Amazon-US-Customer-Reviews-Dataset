from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
from pyspark.sql.functions import col, length, when ## importing everything we might use
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import count, avg, min, max
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import requests
import pandas as pd
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
Matplotlib created a temporary cache directory at /scratch/zalabadleh/job_49109199/matplotlib-ami7s3ph because the default path (/home/jovyan/.cache/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
df = spark.read.csv("/expanse/lustre/projects/uci157/zalabadleh/AmazonReviews/*.tsv", sep="\t", header=True, inferSchema=True)
df.printSchema()  
root
 |-- marketplace: string (nullable = true)
 |-- customer_id: integer (nullable = true)
 |-- review_id: string (nullable = true)
 |-- product_id: string (nullable = true)
 |-- product_parent: integer (nullable = true)
 |-- product_title: string (nullable = true)
 |-- product_category: string (nullable = true)
 |-- star_rating: string (nullable = true)
 |-- helpful_votes: integer (nullable = true)
 |-- total_votes: integer (nullable = true)
 |-- vine: string (nullable = true)
 |-- verified_purchase: string (nullable = true)
 |-- review_headline: string (nullable = true)
 |-- review_body: string (nullable = true)
 |-- review_date: date (nullable = true)

df.count()
58075027
df.describe().show()
+-------+-----------+--------------------+--------------+-------------------+--------------------+--------------------+----------------+-----------------+------------------+------------------+--------+-----------------+--------------------+--------------------+
|summary|marketplace|         customer_id|     review_id|         product_id|      product_parent|       product_title|product_category|      star_rating|     helpful_votes|       total_votes|    vine|verified_purchase|     review_headline|         review_body|
+-------+-----------+--------------------+--------------+-------------------+--------------------+--------------------+----------------+-----------------+------------------+------------------+--------+-----------------+--------------------+--------------------+
|  count|   58075027|            58075027|      58075026|           58075026|            58075026|            58075025|        58073944|         58073934|          58073928|          58073928|58073928|         58073928|            58073848|            58066535|
|   mean|       NULL| 2.709047951711523E7|          NULL|3.042459460646061E9|4.9949849062407655E8|            Infinity|            NULL|4.166902245703097|1.4301775833038193|1.9255968702513113|    NULL|             NULL|                 NaN|            Infinity|
| stddev|       NULL|1.5755498336164678E7|          NULL|2.838510075361177E9|2.8874554637495524E8|                 NaN|            NULL|1.295014662029231|17.962625426391888| 19.19630240225601|    NULL|             NULL|                 NaN|                 NaN|
|    min|         US|                 433|R100007TERQ36I|         0000031852|                 222| Handheld Speaker...|      2011-05-12|                1|                 0|                 0|       N|                N|\tArrived on time...|This movie is ab...|
|    max|         US|            53096592| RZZZZYOFYZ829|         BT00IU6O8K|           999999945|🌴 Vacation On Th...|        Wireless|                5|             41393|             41866|       Y|                Y|🛀⚽️He's likes th...|🛅🚑🚚🚏🚙🚈🚘🚈?...|
+-------+-----------+--------------------+--------------+-------------------+--------------------+--------------------+----------------+-----------------+------------------+------------------+--------+-----------------+--------------------+--------------------+

from pyspark.sql.functions import count, avg, min, max
>>> df.groupBy("product_category") \
      .agg(
          count("*").alias("num_reviews"),
          avg("star_rating").alias("avg_rating"),
          avg("helpful_votes").alias("avg_helpful_votes")
) \
.orderBy("num_reviews", ascending=False) \
.show(10)
+----------------+-----------+------------------+------------------+
|product_category|num_reviews|        avg_rating| avg_helpful_votes|
+----------------+-----------+------------------+------------------+
|        Wireless|    7266941|   3.9254046510079|0.6362107522271063|
|              PC|    6283460|4.1024690218446525|1.1721180687073682|
|            Toys|    4864293|4.2117329280945865| 1.477076730369655|
|          Sports|    4849563| 4.229311795722625| 1.473776296957066|
|       Video DVD|    4690455| 4.328275401853339|2.6382939395005387|
|           Shoes|    4366916| 4.241260422687315|0.8771812876638799|
|    Pet Products|    2643614| 4.143652969003796|1.6528846495744083|
| Office Products|    2642409| 4.072544030844582|1.9013086164935102|
|        Outdoors|    2302392| 4.240021681798756|  1.84732530342357|
|     Video Games|    1786145|4.0599313045693375| 2.260269463005523|
+----------------+-----------+------------------+------------------+
only showing top 10 rows

top_categories = df.filter(col("product_category").isNotNull()) \
    .groupBy("product_category") \
    .count() \
    .orderBy("count", ascending=False) \
    .limit(15)

pdf = top_categories.toPandas()
plt.figure(figsize=(10,6))                                                                                           
plt.barh(pdf["product_category"], pdf["count"])
plt.title("Top 10 Product Categories by Number of Reviews")
plt.xlabel("Number of Reviews")
plt.ylabel("Product Category")
plt.gca().invert_yaxis()
plt.show()

scatter_df = df.select("star_rating", "helpful_votes") \
    .filter(col("star_rating").isNotNull()) \
    .filter(col("helpful_votes").isNotNull()) \
    .sample(0.01)

pdf = scatter_df.toPandas()
plt.figure(figsize=(8,5))
plt.scatter(
    pdf["star_rating"].astype(float),
    pdf["helpful_votes"],
    alpha=0.3
)

plt.title("Star Rating vs Helpful Votes")
plt.xlabel("Star Rating")
plt.ylabel("Helpful Votes")
plt.show()

#Get the active Spark Context and URL
sc = spark.sparkContext

url = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}/executors"

#Fetch the executor data from the API
response = requests.get(url)
executors = response.json()

#Format into a readable DataFrame
executor_df = pd.DataFrame(executors)[['id', 'totalCores', 'maxMemory', 'activeTasks', 'isActive']]
executor_df['maxMemory_GB'] = (executor_df['maxMemory'] / (1024**3)).round(2)
executor_df
