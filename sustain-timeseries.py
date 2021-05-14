import time

import pandas as pd
import os
from prophet import Prophet
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

jars = "" \
       "./jars/mongo-spark-connector_2.12-3.0.1.jar," \
       "./jars/mongo-java-driver-3.12.5.jar," \
       "./jars/bson-4.0.5.jar," \
       "./jars/spark-core_2.12-3.0.1.jar," \
       "./jars/spark-sql_2.12-3.0.1.jar"

# .config("spark.jars", jars) \
SPARK_MASTER = os.environ["SPARK_MASTER"]
DB_NAME = os.environ["DB_NAME"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
SPARK_THREAD_COUNT = os.environ["SPARK_THREAD_COUNT"]
SPARK_EXECUTOR_CORES = os.environ["SPARK_EXECUTOR_CORES"]
SPARK_EXECUTOR_MEMORY = os.environ["SPARK_EXECUTOR_MEMORY"]
SPARK_INITIAL_EXECUTORS = os.environ["SPARK_INITIAL_EXECUTORS"]
SPARK_MIN_EXECUTORS = os.environ["SPARK_MIN_EXECUTORS"]
SPARK_MAX_EXECUTORS = os.environ["SPARK_MAX_EXECUTORS"]
SPARK_BACKLOG_TIMEOUT = os.environ["SPARK_BACKLOG_TIMEOUT"]
SPARK_IDLE_TIMEOUT = os.environ["SPARK_IDLE_TIMEOUT"]

GISJOIN = "GISJOIN"

spark = SparkSession \
    .builder \
    .master(SPARK_MASTER) \
    .appName("COVID-19 Time-series - PySpark") \
    .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
    .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") \
    .config("park.executor.cores", SPARK_EXECUTOR_CORES) \
    .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY) \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.dynamicAllocation.shuffleTracking.enabled", "true") \
    .config("spark.dynamicAllocation.initialExecutors", SPARK_INITIAL_EXECUTORS) \
    .config("spark.dynamicAllocation.minExecutors", SPARK_MIN_EXECUTORS) \
    .config("spark.dynamicAllocation.maxExecutors", SPARK_MAX_EXECUTORS) \
    .config("spark.dynamicAllocation.schedulerBacklogTimeout", SPARK_BACKLOG_TIMEOUT) \
    .config("spark.dynamicAllocation.executorIdleTimeout", SPARK_IDLE_TIMEOUT) \
    .getOrCreate()

print(f'Spark Version: {spark.sparkContext.version}')

mongo_connection_uri = f'mongodb://{DB_HOST}:{DB_PORT}/{DB_NAME}.covid_county_formatted'
print(f'mongo_connection_uri: {mongo_connection_uri}')
df = spark.read.format("mongo").option("uri", mongo_connection_uri).load()

df = df.select(GISJOIN, 'cases', 'deaths', 'date', 'formatted_date')

df.show()

result_schema = StructType([
    StructField("ds", DateType(), True),
    StructField("yhat", DoubleType(), True),
    StructField("yhat_lower", DoubleType(), True),
    StructField("yhat_upper", DoubleType(), True)
])
print('log: result_schema created')


@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def predict(df0):
    # instantiate the model, configure the parameters
    m = Prophet()
    m.fit(df0)
    df0_future = m.make_future_dataframe(periods=365)
    df0_forecast = m.predict(df0_future)

    return df0_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


df_cases = df.select(GISJOIN, 'date', 'cases').withColumnRenamed('date', 'ds').withColumnRenamed('cases', 'y')
print('Showing df_cases')
df_cases.show()

results = (df_cases.groupBy(GISJOIN).apply(predict))

print('log: Showing results')
results.show()

time1 = time.monotonic()
print(f'time1: {time1}')

results.count()

time2 = time.monotonic()
print(f'time2: {time2}')

print(f'Time Taken: {time2 - time1}')
