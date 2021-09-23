import pandas as pd
from prophet import Prophet
from pymongo import MongoClient
from prophet.plot import plot_plotly, plot_components_plotly
import os
import time
import pickle
import numpy as np
from datetime import datetime
import itertools
import dask
from dask.distributed import Client


DASK_URL = "lattice-150:8786"
SINGLE_MODEL = True

df_parent = pd.read_csv('covid_parents_trained.csv')
print(f'df_parent.shape: {df_parent.shape}')


class TrainedParent:
    def __init__(self, gis_join, rmse, changepoint_prior_scale, seasonality_prior_scale):
        self.gis_join = gis_join
        self.rmse = rmse
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale

    def __str__(self):
        return f'{self.gis_join}: (rmse={self.rmse}, changepoint_prior_scale={self.changepoint_prior_scale}, seasonality_prior_scale={self.seasonality_prior_scale})'


trained_parents_map = {}
for i, row in df_parent.iterrows():
    gis_join = row['GISJOIN']
    rmse = row['rmse']
    changepoint_prior_scale = row['changepoint_prior_scale']
    seasonality_prior_scale = row['seasonality_prior_scale']
    trained_parents_map[gis_join] = (TrainedParent(gis_join,
                                                   rmse, changepoint_prior_scale, seasonality_prior_scale))

print(len(trained_parents_map.keys()))

# Load Child Parent Map
parent_child_map = pickle.load(open('pickles/parent_child_map.pkl', 'rb'))

df_clusters = pd.read_csv('./clusters-covid.csv')
print(df_clusters.head())

parents_pickle = set(parent_child_map.keys())
parents_csv = set(df_clusters[df_clusters['is_parent'] == 1]['GISJOIN'])

# print(f'parents_pickle: {parents_pickle}')
# print(f'parents_csv: {parents_csv}')

print(f'Difference between two sets of parents: {len(parents_pickle - parents_csv)}')

# child GISJOIN to sample_percent map
child_sample_perc_map = {}
for i, row in df_clusters.iterrows():
    is_parent = row['is_parent']
    sample_percent = row['sample_percent']
    if not is_parent and sample_percent <= 0.15:
        gis_join = row['GISJOIN']

        child_sample_perc_map[gis_join] = sample_percent

children_list = list(child_sample_perc_map.keys())
no_of_children = len(children_list)
no_of_parents = len(trained_parents_map.keys())

# assert no_of_children == (df_clusters.shape[0] - no_of_parents)
print(f'no_of_children: {no_of_children}')
print(f'no_of_parents: {no_of_parents}')

db = MongoClient("lattice-100", 27018)
collection = 'covid_county_formatted'


def get_df_by_gis_join(gis_join, sample_percent=1.0):
    print(gis_join, end=' ')
    cursor = db.sustaindb[collection].aggregate([{"$match": {"GISJOIN": gis_join}}])
    df = pd.DataFrame(list(cursor))[['date', 'cases']]
    df.columns = ['ds', 'y']
    return df.sample(frac=sample_percent)

def predict_transfer(df_train, parent_trained):
    time1 = time.monotonic()
    # initilaize model with hyperparameters from parent model
    m = Prophet(
        seasonality_prior_scale = parent_trained.seasonality_prior_scale,
        changepoint_prior_scale = parent_trained.changepoint_prior_scale,
    )
    m.fit(df_train, algorithm='LBFGS')
    df_train_future = m.make_future_dataframe(periods=300, freq='H')
    df_train_forecast = m.predict(df_train_future)
    time2 = time.monotonic()

    return m, df_train_future, df_train_forecast, (time2 - time1)


def predict_transfer_task(df_train, gis_join, parent_trained):
    m, df_train_future, df_train_forecast, time_taken = predict_transfer(df_train, parent_trained)
    return gis_join, time_taken


children_dfs_map = {}

for gis_join, sample_percent in child_sample_perc_map.items():
    children_dfs_map[gis_join] = get_df_by_gis_join(gis_join, sample_percent)

# pickle.dump(child_sample_perc_map, open('pickles/child_sample_perc_map.pkl', 'wb'))
pickle.dump(children_dfs_map, open('pickles/children_dfs_map.pkl', 'wb'))
# children_dfs_map = pickle.load(open('pickles/children_dfs_map.pkl', 'rb'))

print(len(children_list))
print(len(children_dfs_map.keys()))

client = Client(DASK_URL)

time1 = time.monotonic()

counter = 1
lazy_results = []
for parent, children in parent_child_map.items():
    try:
        parent_trained_  = trained_parents_map[parent]
        print(parent_trained_)
        for child in children:
            # check if the child falls under < 0.15 sampling percentage
            if child in children_dfs_map:
                child_df = children_dfs_map[child]
                lazy_result = dask.delayed(predict_transfer_task)(child_df, gis_join, parent_trained_)
                lazy_results.append(lazy_result)
                if SINGLE_MODEL:
                    break
    except Exception as e:
        print(f'Error on {gis_join}')
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
    if counter % 100 == 0:
        print(counter, end=', ')
    counter += 1

futures = dask.persist(*lazy_results)  # trigger computation in the background
results = dask.compute(*futures)
print('Printing first 5 results')
print(results[:5])

time2 = time.monotonic()
print(f'Time taken (dataset=COVID-19, childModels 0-15%: {time2 - time1} s')

# Write to CSV
if not SINGLE_MODEL:
    print('Writing results to csv')

    gis_joins = []
    times = []

    for r, t in results:
        gis_joins.append(r)
        times.append(t)

    times_0_15_df = pd.DataFrame(zip(gis_joins, times), columns=['GISJOIN', 'time'])
    # times_0_15_df.to_csv('covid-19-child_training_tl_times_0_15.csv', index=False)
    print(times_0_15_df.head())

