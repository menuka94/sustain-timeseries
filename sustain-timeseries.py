import pandas as pd
from prophet import Prophet
from pymongo import MongoClient
from prophet.plot import plot_plotly, plot_components_plotly

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

from pyspark.sql import SparkSession 

spark = SparkSession.builder.master("local").getOrCreate() 
print(f'Spark Version: {spark.sparkContext.version}')

df = spark.read.format('json').load('./covid_county.json')

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

results.show()
