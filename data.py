# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:19:25 2019

@author: Mickie(Ziyi) Xu
"""

from influxdb import InfluxDBClient
import matplotlib.pyplot as plt
import pandas as pd
import time
import thread
import statsmodels.api as sm

def get_data():
    client = InfluxDBClient(host='172.16.248.177', port=8086, username='root', database='telegraf')
    result = client.query("select LAST(usage_user) from cpu where cpu = 'cpu-total';")
    return result

def get_datas(data1, data2):
    client = InfluxDBClient(host='172.16.248.177', port=8086, username='root', database='telegraf')
    result = client.query("select LAST(usage_user) from cpu where cpu = 'cpu-total' and time >= '%s' AND time <= '%s' GROUP BY time(10s),* fill(10) LIMIT 360 SLIMIT 1;" %(data1, data2))
    return result

def update_data():
    data_cpu = get_data()
    if data is NULL:
        data = []
        data.append(data_cpu)
    else:
        data.append(data_cpu)
    return data

def updata_datas():
    data1 = input("input start time(example:2018-12-22T14:40:00Z):")
    data2 = input("input end time(example:2018-12-22T14:40:30Z):")
    data_cpu = get_datas(data1, data2)
    return data_cpu

def cpu_ARIMA(p, d, q, t):
    data = updata_cpu()
    new_data = data.diff(d)
    model = sm.tsa.ARMA(new_data,(p,q)).fit()
    predict_data = model.predict(t)

'''
cpu_ARIMA(1, 1, 1, 6)
'''

def run_data(threadname, x):
    y = 1
    while y < 10:
        time.sleep(x)
        time = time.ctime()
        data = data_update()
        print(threadname,':', time,',', data[-1])
    
def predict_data(threadname, x):
    y = 1
    while y < 10:
        time.sleep(x)
        time = time.ctime()
        data = data_update()
        new_data = data[-60:].diff(1)
        model = sm.tsa.ARMA(new_data,(1,1)).fit()
        predict_data = model.predict(6)
        print(threadname,':', time,',', predict_data)

'''
try:
   thread.start_new_thread(run_data, ("Thread-1", 10, ) )
   thread.start_new_thread(predict_data, ("Thread-2", 10, ) )
except:
   print("Error: oh my god, we are dead!")

while 1:
    pass
'''