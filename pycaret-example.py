#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 19:01:31 2021

@author: ugur
"""

# read csv file
import pandas as pd
import numpy as np
import datetime 
# import the regression module
from pycaret.regression import *

def convert_date(date_string):
        date_string=date_string.split(".")
        new_date=date_string[1]+"/"+date_string[0]+"/"+date_string[2]
        print("new_date", new_date)
        return new_date
    
    
# data = pd.read_csv('AirPassengers.csv')
# data['Date'] = pd.to_datetime(data['Date'])
# data.head()


df1= pd.read_csv('https://raw.githubusercontent.com/ozanerturk/covid19-turkey-api/master/dataset/timeline.csv', index_col=False)
dates=df1[df1.columns[10]].tolist()
dates=[x.replace("/",".") for x in dates]
start_convert_date = convert_date(dates[0])
end_convert_date = convert_date(dates[-1])
 
dates_range=pd.date_range(start=start_convert_date, end=end_convert_date)
patients=df1[df1.columns[0]].tolist()

df = pd.DataFrame()

df["Date"] = dates_range
df["patients"] = patients

# extract day, month and year from dates
df['Day'] = [i.day for i in df['Date']]
df['Month'] = [i.month for i in df['Date']]
df['Year'] = [i.year for i in df['Date']]

# create a sequence of numbers
df['Series'] = np.arange(1,len(df)+1)

df.drop(["Date"], axis=1, inplace=True)
df = df[['Series', 'Year', 'Month', 'patients']] 
# check the head of the dataset

train_pct_index = int(0.9 * len(df))
train, test = df[:train_pct_index], df[train_pct_index:]

# initialize setup
s = setup(data = train, test_data = test, target = 'patients', 
          fold_strategy = 'timeseries', numeric_features = ['Year','Month','Series'], 
          fold = 3, session_id = 123)

best = compare_models(sort = 'MAE')

prediction_holdout = predict_model(best);
# generate predictions on the original dataset
predictions = predict_model(best, data=df)

final_best = finalize_model(best)

prediction_start_date = dates_range[-1] + datetime.timedelta(days = 1)
prediction_end_date = prediction_start_date + datetime.timedelta(days = 6)

future_dates = pd.date_range(start = prediction_start_date, end = prediction_end_date)
future_df = pd.DataFrame()
future_df['Day'] = [i.day for i in future_dates]
future_df['Month'] = [i.month for i in future_dates]
future_df['Series'] = np.arange(df['Series'].iloc[-1]+1,(df['Series'].iloc[-1]+1+len(future_dates)))
future_df['Year'] = [i.year for i in future_dates]    

predictions_future = predict_model(final_best, data=future_df)
print(predictions_future.head())



