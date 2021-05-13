import time

import pandas as pd
from pymongo import MongoClient
import json
import requests
from prophet import Prophet
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

# .config("spark.jars", "org.mongodb:bson:4.0.5") \

jars = "" \
       "./jars/mongo-spark-connector_2.12-3.0.1.jar," \
       "./jars/mongo-java-driver-3.12.5.jar," \
       "./jars/bson-4.0.5.jar," \
       "./jars/spark-core_2.12-3.0.1.jar," \
       "./jars/spark-sql_2.12-3.0.1.jar"

spark = SparkSession \
    .builder \
    .master("spark://lattice-100:8079") \
    .appName("COVID-19 Time-series - PySpark") \
    .config("spark.jars", jars) \
    .getOrCreate()
sqlContext = SQLContext(spark.sparkContext)

print(f'Spark Version: {spark.sparkContext.version}')


# -----------------------------------------------------------------

# Auxiliar functions
# Pandas Types -> Sparks Types
def equivalent_type(f):
    if f == 'datetime64[ns]':
        return DateType()
    elif f == 'int64':
        return LongType()
    elif f == 'int32':
        return IntegerType()
    elif f == 'float64':
        return FloatType()
    else:
        return StringType()


def define_structure(string, format_type):
    try:
        typo = equivalent_type(format_type)
    except:
        typo = StringType()
    return StructField(string, typo)


# Given pandas dataframe, it will return a spark's dataframe
def pandas_to_spark(df_pandas):
    columns = list(df_pandas.columns)
    types = list(df_pandas.dtypes)
    struct_list = []
    for column, typo in zip(columns, types):
        struct_list.append(define_structure(column, typo))
    p_schema = StructType(struct_list)
    return sqlContext.createDataFrame(df_pandas, p_schema)


# -----------------------------------------------------------------


# db = MongoClient('lattice-100', 27018)
# collection = 'covid_county_formatted'
# cursor = db.sustaindb[collection].find()
# df_pd = pd.DataFrame(list(cursor))
#
# df = pandas_to_spark(df_pd)

url = 'https://bitbucket.org/menuka94/sustain-covid-county-data/raw/357e0f6b964d5c6ece880846c4d1caad1f88a65c/covid_county.json'
# from pyspark import SparkFiles
# spark.sparkContext.addFile(url)
# # df = spark.read.format('json').load('')
# df = spark.read.json(SparkFiles.get('covid_county.json'))

r = requests.get(url)
df = sqlContext.createDataFrame([json.loads(line) for line in r.iter_lines()])

df = df.select('GISJOIN', 'cases', 'deaths', 'date', 'formatted_date')

df.show()

result_schema = StructType([
    StructField("ds", DateType(), True),
    StructField("yhat", DoubleType(), True),
    StructField("yhat_lower", DoubleType(), True),
    StructField("yhat_upper", DoubleType(), True)
])
print('log: result_schema created')


@pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
def temp(df0):
    # instantiate the model, configure the parameters
    m = Prophet()
    m.fit(df0)
    df0_future = m.make_future_dataframe(periods=365)
    df0_forecast = m.predict(df0_future)

    return df0_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


df_cases = df.select('GISJOIN', 'date', 'cases').withColumnRenamed('date', 'ds').withColumnRenamed('cases', 'y')
print('Showing df_cases')
df_cases.show()

results = (df_cases.groupBy('GISJOIN').apply(temp))

print('log: Showring results')
results.show()

time1 = time.monotonic()
print(f'time1: {time1}')

results.count()

time2 = time.monotonic()
print(f'time2: {time2}')

print(f'Time Taken: {time2 - time1}')
