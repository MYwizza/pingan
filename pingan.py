# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:34:43 2018

@author: miya
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt  

train_path=r'C:\Users\miya\Documents\GitHub\pingan\PINGAN-2018-train_demo.csv'
train=pd.read_csv(train_path)
train.head()
train1=[]

for i in [2]:
    
    item=train.loc[train['TERMINALNO']==i]
    #时间特征
    item['TIME_Tsfd']=item['TIME'].apply(lambda x:datetime.datetime.fromtimestamp(x))
    item['TIME_Tsfd']=pd.to_datetime(item['TIME_Tsfd'])
    item['Month']=item['TIME_Tsfd'].dt.month
    item['hour']=item['TIME_Tsfd'].dt.hour
    item['week_of_year']=item['TIME_Tsfd'].dt.weekofyear
    item['day_of_week']=item['TIME_Tsfd'].dt.dayofweek
    trips_data_num=item['TERMINALNO'].shape[0]
    
    hour_state=np.zeros([24,1])
    for hour in range(24):
        hour_state[hour]=item.loc[item['hour']==hour].shape[0]/trips_data_num
        
    #trip个数特征
    trip_num=item['TRIP_ID'].unique().shape[0]
    
    #速度特征
    average_speed=item.groupby('TRIP_ID')['SPEED'].mean().mean()
    max_speed=item['SPEED'].max()
    trip_average_speed_max=item.groupby('TRIP_ID')['SPEED'].mean().max()
    trip_max_speed_average=item.groupby('TRIP_ID')['SPEED'].max().mean()
     
    #加速度特征
    posi_trip_acceleration_max=[]
    posi_trip_acceleration_mean=[]
    nega_trip_acceleration_max=[]
    nega_trip_acceleration_mean=[]
    
    for trip in item['TRIP_ID'].unique():
        trip_item=item.loc[item['TRIP_ID']==trip]
        acceleration=[]
        for trip_speed in range(trip_item.shape[0]):
            acceleration.append(trip_item['SPEED'].iloc[trip_speed]-trip_item['SPEED'].iloc[trip_speed-1])
        if acceleration[1:]:
            posi_trip_acceleration_max.append(max(acceleration[1:]))
            posi_trip_acceleration_mean.append(sum(acceleration[1:])/len(acceleration[1:]))
            nega_trip_acceleration_max.append(min(acceleration[1:]))
    posi_acceleration_max=max(posi_trip_acceleration_max)
    neg_acceleration_max=min(nega_trip_acceleration_max)
#    acceleration_max_mean=sum(posi_trip_acceleration_max)/len(posi_trip_acceleration_max)
#    acceleration_mean=sum(trip_acceleration_mean)/len(trip_acceleration_mean)
#    acceleration_mean_max=max(trip_acceleration_mean)
    
    feature=[trip_num,average_speed,max_speed,trip_average_speed_max,trip_max_speed_average,float(hour_state[0]),float(hour_state[1]),
             float(hour_state[2]),float(hour_state[3]),float(hour_state[4]),float(hour_state[5])
        ,float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10]),float(hour_state[11])
        ,float(hour_state[12]),float(hour_state[13]),float(hour_state[14]),float(hour_state[15]),float(hour_state[16]),float(hour_state[17])
        ,float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23])]
    
    train1.append(feature)
train1=pd.DataFrame(train1)
feature_name=['trip_num','average_speed','max_speed','trip_average_speed_max','trip_max_speed_average',
              'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17',
              'h18','h19','h20','h21','h22','h23']
train1.columns=feature_name