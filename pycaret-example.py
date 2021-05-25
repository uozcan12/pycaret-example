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
from pycaret.regression import *

def convert_date(date_string):
        date_string=date_string.split(".")
        new_date=date_string[1]+"/"+date_string[0]+"/"+date_string[2]
        print("new_date", new_date)
        return new_date
    
def save_patients_model(column_name, dates_range, model_name):
    df = pd.DataFrame()

    df["Date"] = dates_range
    df[model_name] = column_name
    
    # extract day, month and year from dates
    df['Day'] = [i.day for i in df['Date']]
    df['Month'] = [i.month for i in df['Date']]
    df['Year'] = [i.year for i in df['Date']]
    
    # create a sequence of numbers
    df['Series'] = np.arange(1,len(df)+1)
    
    df.drop(["Date"], axis=1, inplace=True)
    df = df[['Series','Day' ,'Year', 'Month', model_name]] 
    # check the head of the dataset
    
    train_pct_index = int(0.9 * len(df))
    train, test = df[:train_pct_index], df[train_pct_index:]
    
    # initialize setup
    s = setup(data = train, test_data = test, target = model_name, 
              fold_strategy = 'timeseries', numeric_features = ['Day','Year','Month','Series'], 
              fold = 3, session_id = 123)
    
    best = compare_models(sort = 'MAE')
    
    created_model = create_model(best)
    tuned_model = tune_model(best)
    
    #prediction_holdout = predict_model(best);
    # generate predictions on the original dataset
    #predictions = predict_model(best, data=df)
    predictions = predict_model(tuned_model, data=df)
    final_best = finalize_model(best)
    
    save_model(final_best,f'{model_name}_final_model')
    
    prediction_start_date = dates_range[-1] + datetime.timedelta(days = 1)
    prediction_end_date = prediction_start_date + datetime.timedelta(days = 6)
    
    future_dates = pd.date_range(start = prediction_start_date, end = prediction_end_date)
    future_df = pd.DataFrame()
    future_df['Day'] = [i.day for i in future_dates]
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Series'] = np.arange(df['Series'].iloc[-1]+1,(df['Series'].iloc[-1]+1+len(future_dates)))
    future_df['Year'] = [i.year for i in future_dates]    
    
    predictions_future = predict_model(final_best, data=future_df)
    return predictions_future


def pycaret_patients_seven_days_prediction(patients,dates_range, model, model_name):
    
    df = pd.DataFrame()
    
    df["Date"] = dates_range
    df[model_name] = patients
    
    # extract day, month and year from dates
    df['Day'] = [i.day for i in df['Date']]
    df['Month'] = [i.month for i in df['Date']]
    df['Year'] = [i.year for i in df['Date']]
    
    # create a sequence of numbers
    df['Series'] = np.arange(1,len(df)+1)
    
    df.drop(["Date"], axis=1, inplace=True)
    df = df[['Series','Day' ,'Year', 'Month', model_name]] 
    # check the head of the dataset
    
    train_pct_index = int(0.9 * len(df))
    train, test = df[:train_pct_index], df[train_pct_index:]
    
    prediction_start_date = dates_range[-1] + datetime.timedelta(days = 1)
    prediction_end_date = prediction_start_date + datetime.timedelta(days = 6)
    
    future_dates = pd.date_range(start = prediction_start_date, end = prediction_end_date)
    future_df = pd.DataFrame()
    future_df['Day'] = [i.day for i in future_dates]
    future_df['Month'] = [i.month for i in future_dates]
    future_df['Series'] = np.arange(df['Series'].iloc[-1]+1,(df['Series'].iloc[-1]+1+len(future_dates)))
    future_df['Year'] = [i.year for i in future_dates]    
    
    predictions_future = predict_model(model, data=future_df)
    predictions_future = predictions_future.astype(int)
    return predictions_future
    

df1= pd.read_csv('https://raw.githubusercontent.com/ozanerturk/covid19-turkey-api/master/dataset/timeline.csv', index_col=False)
dates=df1[df1.columns[10]].tolist()
dates=[x.replace("/",".") for x in dates]
start_convert_date = convert_date(dates[0])
end_convert_date = convert_date(dates[-1])
 
dates_range=pd.date_range(start=start_convert_date, end=end_convert_date)

##########################################################
patients = df1[df1.columns[0]].tolist()
seven_days_prediction_patients = save_patients_model(patients, dates_range, "patients") 
#model = load_model("patients_final_model")
#seven_days_prediction_patients = pycaret_patients_seven_days_prediction(patients,dates_range, model, "patients")
print(seven_days_prediction_patients)

totalPatients = df1[df1.columns[1]].tolist()
seven_days_prediction_totalPatients = save_patients_model(totalPatients, dates_range, "totalPatients")
#model = load_model("totalPatients_final_model")
#seven_days_prediction_totalPatients = pycaret_patients_seven_days_prediction(totalPatients,dates_range, model,"totalPatients")
print(seven_days_prediction_totalPatients)

deaths = df1[df1.columns[2]].tolist()
seven_days_prediction_deaths = save_patients_model(deaths, dates_range, "deaths")
#model = load_model("deaths_final_model")
#seven_days_prediction_deaths = pycaret_patients_seven_days_prediction(deaths,dates_range, model, "deaths")
print(seven_days_prediction_deaths)

totalDeaths = df1[df1.columns[3]].tolist()
seven_days_prediction_totalDeaths = save_patients_model(totalDeaths, dates_range, "totalDeaths")
#model = load_model("totalDeaths_final_model")
#seven_days_prediction_totalDeaths = pycaret_patients_seven_days_prediction(totalDeaths, dates_range, model, "totalDeaths")
print(seven_days_prediction_totalDeaths)

recovered = df1[df1.columns[4]].tolist()
seven_days_prediction_recovered = save_patients_model(recovered, dates_range, "recovered")
#model = load_model("recovered_final_model")
#seven_days_prediction_recovered = pycaret_patients_seven_days_prediction(recovered, dates_range, model, "recovered")
print(seven_days_prediction_recovered)

totalRecovered = df1[df1.columns[5]].tolist()
seven_days_prediction_totalRecovered = save_patients_model(totalRecovered, dates_range, "totalRecovered")
#model = load_model("totalRecovered_final_model")
#seven_days_prediction_totalRecovered = pycaret_patients_seven_days_prediction(totalRecovered, dates_range, model, "totalRecovered")
print(seven_days_prediction_totalRecovered)

tests = df1[df1.columns[8]].tolist()
seven_days_prediction_tests = save_patients_model(tests, dates_range, "tests")
#model = load_model("tests_final_model")
#seven_days_prediction_totalDeaths = pycaret_patients_seven_days_prediction(totalDeaths, dates_range, model, "totalDeaths")
print(seven_days_prediction_tests)

totalTests = df1[df1.columns[9]].tolist()
seven_days_prediction_totalTests = save_patients_model(totalTests, dates_range, "totalTests")
#model = load_model("totalTests_final_model")
#seven_days_prediction_totalTests = pycaret_patients_seven_days_prediction(totalTests, dates_range, model, "totalTests")
print(seven_days_prediction_totalTests)


cases = df1[df1.columns[13]].tolist()
seven_days_prediction_cases = save_patients_model(cases, dates_range, "cases")
#model = load_model("cases_final_model")
#seven_days_prediction_cases = pycaret_patients_seven_days_prediction(cases, dates_range, model, "cases")
print(seven_days_prediction_cases)



"""
"patients", 		0
"totalPatients",    1
"deaths",    2
"totalDeaths",  3
"recovered",   4
"totalRecovered",  5
"totalIntubated",  6
"totalIntensiveCare", 7
"tests", 8
"totalTests", 9
"date", 10
"critical",  11
"pneumoniaPercent", 12
"cases" 13

"""


