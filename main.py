# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:03:26 2018

@author: GTAdmin
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:34:43 2018

@author: miya
"""

import pandas as pd
import numpy as np
import datetime

start_all=datetime.datetime.now()
print('start time:',start_all)
#train_path=r'D:\LJK\pingan\PINGAN-2018-train_demo.csv'
path_train="/data/dm/train.csv"
#path_train=r'C:\Users\miya\Documents\GitHub\pingan\PINGAN-2018-train_demo.csv'
path_test="/data/dm/test.csv"
path_result_out="model/pro_result.csv"
#path_result_out=r'C:\Users\miya\Documents\GitHub\pingan\pro_result.csv'

train=pd.read_csv(path_train)
train.head()
train1=[]

def mean_max(item,speed,direction,height,func_select):
    posi_trip_acceleration_max=[]#正加速度最大值
    posi_trip_acceleration_mean=[]#正加速度的平均值
    nega_trip_acceleration_max=[]#负加速度最大值
    nega_trip_acceleration_mean=[]#负加速度的平均值
    acceleration_mean=[]#总加速度平均值
    posi_trip_acceleration=[]
    nega_trip_acceleration=[]
    
    for trip in item['TRIP_ID'].unique():
        trip_item=item.loc[item['TRIP_ID']==trip]
        acceleration=[]
        posi_acceleration=[]
        nega_acceleration=[]
        direction_a=[]
        acceleration_a=[]
        height_a=[]
#        posi_dir_inc=[]
#        posi_dir_dec=[]
#        nega_dir_inc=[]
#        nega_dir_dec=[]
        
        for trip_speed in range(trip_item.shape[0]):
            acceleration_m=trip_item[speed].iloc[trip_speed]-trip_item[speed].iloc[trip_speed-1]
            direction_m=trip_item[direction].iloc[trip_speed]-trip_item[direction].iloc[trip_speed-1]
            height_m=trip_item[height].iloc[trip_speed]-trip_item[height].iloc[trip_speed-1]

            if direction_m>180:
                direction_m=direction_m-360
            elif direction_m<=-180:
                direction_m=360+direction_m
                
            acceleration_a.append(acceleration_m)
            direction_a.append(direction_m)
            height_a.append(height_m)
            
#            if trip_speed:
#                if direction_m>=0 & acceleration_m>0:
#                    posi_dir_inc.append(direction_m/acceleration_m)
#                elif direction_m>=0 & acceleration_m<0:
#                    posi_dir_dec.append(direction_m/acceleration_m)
#                elif direction_m<=0 & acceleration_m>0:
#                    nega_dir_inc.append(direction_m/acceleration_m)
#                elif direction_m<=0 & acceleration_m<0:
#                    nega_dir_dec.append(direction_m/acceleration_m)
            
            if func_select==1:
                acceleration=acceleration_a
            elif func_select==2:
                acceleration=direction_a
            elif func_select==3:
                acceleration=height_a
         
        for acc in acceleration[1:]:
            if acc>0:
                posi_acceleration.append(acc)
            elif acc<0:
                nega_acceleration.append(acc)
        
        if acceleration[1:]:
            acceleration_mean.append(sum(acceleration[1:])/len(acceleration[1:]))
        
        if posi_acceleration:
            posi_trip_acceleration_max.append(max(posi_acceleration))
            posi_trip_acceleration_mean.append(sum(posi_acceleration[1:])/len(posi_acceleration))
        if nega_acceleration:
            nega_trip_acceleration_max.append(min(nega_acceleration))
            nega_trip_acceleration_mean.append(sum(nega_acceleration[1:])/len(nega_acceleration))
        posi_trip_acceleration.append(posi_acceleration)
        nega_trip_acceleration.append(nega_acceleration)
    am,pta_mean,pta_mean_max,nta_mean,nta_mean_max,psa_max,nsa_max=0,0,0,0,0,0,0   
    if acceleration_mean:
        am=sum(acceleration_mean)/len(acceleration_mean)#总加速度平均值

    if posi_trip_acceleration_mean:
        pta_mean=sum(posi_trip_acceleration_mean)/len(posi_trip_acceleration_mean)#正加速度平均值

    if posi_trip_acceleration_mean:
        pta_mean_max=max(posi_trip_acceleration_mean)#每段trip正加速度平均值的最大值

        
    if nega_trip_acceleration_mean:
        nta_mean=sum(nega_trip_acceleration_mean)/len(nega_trip_acceleration_mean)#负加速度平均值
        nta_mean_max=min(nega_trip_acceleration_mean)#每段trip负加速度平均值的最大值
    if posi_trip_acceleration_max:
        psa_max=max(posi_trip_acceleration_max)#正加速度最大值
    if nega_trip_acceleration_max:
        nsa_max=min(nega_trip_acceleration_max)#负加速度最大值
    return am,pta_mean,pta_mean_max,nta_mean,nta_mean_max,psa_max,nsa_max

for i in train['TERMINALNO'].unique():
    
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
    
    #callstate特征
    num_of_state0=item.loc[item['CALLSTATE']==0].shape[0]/trips_data_num
    num_of_state1=item.loc[item['CALLSTATE']==1].shape[0]/trips_data_num   
    num_of_state2=item.loc[item['CALLSTATE']==2].shape[0]/trips_data_num  
    num_of_state3=item.loc[item['CALLSTATE']==3].shape[0]/trips_data_num
    num_of_state4=item.loc[item['CALLSTATE']==4].shape[0]/trips_data_num
    
    #速度特征
    average_speed=item.groupby('TRIP_ID')['SPEED'].mean().mean()
    max_speed=item['SPEED'].max()
    trip_average_speed_max=item.groupby('TRIP_ID')['SPEED'].mean().max()
    trip_max_speed_average=item.groupby('TRIP_ID')['SPEED'].max().mean()
     
    #加速度特征
    acc_m,pt_acc_mean,pt_acc_mean_max,nt_acc_mean,nt_acc_mean_max,ps_acc_max,ns_acc_max=mean_max(item,'SPEED','DIRECTION','HEIGHT',1)
    #方向特征
    dir_m,pt_dir_mean,pt_dir_mean_max,nt_dir_mean,nt_dir_mean_max,ps_dir_max,ns_dir_max=mean_max(item,'SPEED','DIRECTION','HEIGHT',2)
    #高度特征
    hgt_m,pt_hgt_mean,pt_hgt_mean_max,nt_hgt_mean,nt_hgt_mean_max,ps_hgt_max,ns_hgt_max=mean_max(item,'SPEED','DIRECTION','HEIGHT',3)
 
    
    #目标
    target=item['Y'].iloc[0]
    feature=[i,trip_num,average_speed,max_speed,trip_average_speed_max,trip_max_speed_average,
             acc_m,pt_acc_mean,pt_acc_mean_max,nt_acc_mean,nt_acc_mean_max,ps_acc_max,ns_acc_max,
             pt_dir_mean,pt_dir_mean_max,nt_dir_mean,nt_dir_mean_max,ps_dir_max,ns_dir_max,
             hgt_m,pt_hgt_mean,pt_hgt_mean_max,nt_hgt_mean,nt_hgt_mean_max,ps_hgt_max,ns_hgt_max,
             num_of_state0,num_of_state1,num_of_state2,num_of_state3,num_of_state4,
             float(hour_state[0]),float(hour_state[1]),float(hour_state[2]),float(hour_state[3]),float(hour_state[4]),float(hour_state[5])
        ,float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10]),float(hour_state[11])
        ,float(hour_state[12]),float(hour_state[13]),float(hour_state[14]),float(hour_state[15]),float(hour_state[16]),float(hour_state[17])
        ,float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23]),
        target]
    
    train1.append(feature)
train1=pd.DataFrame(train1)
feature_name=['item','trip_num','average_speed','max_speed','trip_average_speed_max','trip_max_speed_average',
              'avetage_acc','posi_acc_mean','posi_acc_mean_max','nega_acc_mean','nega_acc_mean_max','posi_acc_max','nega_acc_max',
              'posi_dir_mean','posi_dir_mean_max','nega_dir_mean','nega_dir_mean_max','posi_dir_max','nega_dir_max',
              'average_hgt','posi_hgt_mean','posi_hgt_mean_max','nega_hgt_mean','nega_hgt_mean_max','posi_hgt_max','nega_hgt_max',
              'num_of_state0','num_of_state1','num_of_state2','num_of_state3','num_of_state4',
              'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17',
              'h18','h19','h20','h21','h22','h23','target']
train1.columns=feature_name

#建立xgb模型
import xgboost as xgb
from sklearn.model_selection import train_test_split

do_not_use_for_training=['item','target']
feature_names=[f for f in train1.columns if f not in do_not_use_for_training]
y=np.array(train1['target'])
#
#对train进行训练数据和测试数据分割
xtr,xv,ytr,yv=train_test_split(train1[feature_names].values,y,test_size=0.2,random_state=1987)
dtrain=xgb.DMatrix(xtr,label=ytr)
dvalid=xgb.DMatrix(xv,label=yv)
#dtest=xgb.DMatrix(test[feature_names].values)
watch=[(dtrain,'train'),(dvalid,'valide')]
xgb_pars={'min_child_weight':50,'eta':0.3,'colsample_bytree':0.3,'max_depth':10,'subsample':0.8,
          'lambda':1.,'nthread':-1,'booster':'gbtree','silent':1,'eval_metric':'rmse','objective':'reg:linear'}
model=xgb.train(xgb_pars,dtrain,15,watch,early_stopping_rounds=2,maximize=False,verbose_eval=1)
print('modeling RMSLE %.5f'% model.best_score)


#测试数据
test=pd.read_csv(path_test)
test1=[]

for i in test['TERMINALNO'].unique():
    
    item=test.loc[train['TERMINALNO']==i]
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
    
    #callstate特征
    num_of_state0=item.loc[item['CALLSTATE']==0].shape[0]/trips_data_num
    num_of_state1=item.loc[item['CALLSTATE']==1].shape[0]/trips_data_num   
    num_of_state2=item.loc[item['CALLSTATE']==2].shape[0]/trips_data_num  
    num_of_state3=item.loc[item['CALLSTATE']==3].shape[0]/trips_data_num
    num_of_state4=item.loc[item['CALLSTATE']==4].shape[0]/trips_data_num
    
    #速度特征
    average_speed=item.groupby('TRIP_ID')['SPEED'].mean().mean()
    max_speed=item['SPEED'].max()
    trip_average_speed_max=item.groupby('TRIP_ID')['SPEED'].mean().max()
    trip_max_speed_average=item.groupby('TRIP_ID')['SPEED'].max().mean()
     
    #加速度特征
    acc_m,pt_acc_mean,pt_acc_mean_max,nt_acc_mean,nt_acc_mean_max,ps_acc_max,ns_acc_max=mean_max(item,'SPEED','DIRECTION','HEIGHT',1)
    #方向特征
    dir_m,pt_dir_mean,pt_dir_mean_max,nt_dir_mean,nt_dir_mean_max,ps_dir_max,ns_dir_max=mean_max(item,'SPEED','DIRECTION','HEIGHT',2)
    #高度特征
    hgt_m,pt_hgt_mean,pt_hgt_mean_max,nt_hgt_mean,nt_hgt_mean_max,ps_hgt_max,ns_hgt_max=mean_max(item,'SPEED','DIRECTION','HEIGHT',3)
 
    
    #目标
    target=-1
    feature=[i,trip_num,average_speed,max_speed,trip_average_speed_max,trip_max_speed_average,
             acc_m,pt_acc_mean,pt_acc_mean_max,nt_acc_mean,nt_acc_mean_max,ps_acc_max,ns_acc_max,
             pt_dir_mean,pt_dir_mean_max,nt_dir_mean,nt_dir_mean_max,ps_dir_max,ns_dir_max,
             hgt_m,pt_hgt_mean,pt_hgt_mean_max,nt_hgt_mean,nt_hgt_mean_max,ps_hgt_max,ns_hgt_max,
             num_of_state0,num_of_state1,num_of_state2,num_of_state3,num_of_state4,
             float(hour_state[0]),float(hour_state[1]),float(hour_state[2]),float(hour_state[3]),float(hour_state[4]),float(hour_state[5])
        ,float(hour_state[6]),float(hour_state[7]),float(hour_state[8]),float(hour_state[9]),float(hour_state[10]),float(hour_state[11])
        ,float(hour_state[12]),float(hour_state[13]),float(hour_state[14]),float(hour_state[15]),float(hour_state[16]),float(hour_state[17])
        ,float(hour_state[18]),float(hour_state[19]),float(hour_state[20]),float(hour_state[21]),float(hour_state[22]),float(hour_state[23]),
        target]
    
    test1.append(feature)
test1=pd.DataFrame(test1)
feature_name=['item','trip_num','average_speed','max_speed','trip_average_speed_max','trip_max_speed_average',
              'avetage_acc','posi_acc_mean','posi_acc_mean_max','nega_acc_mean','nega_acc_mean_max','posi_acc_max','nega_acc_max',
              'posi_dir_mean','posi_dir_mean_max','nega_dir_mean','nega_dir_mean_max','posi_dir_max','nega_dir_max',
              'average_hgt','posi_hgt_mean','posi_hgt_mean_max','nega_hgt_mean','nega_hgt_mean_max','posi_hgt_max','nega_hgt_max',
              'num_of_state0','num_of_state1','num_of_state2','num_of_state3','num_of_state4',
              'h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17',
              'h18','h19','h20','h21','h22','h23','target']
test1.columns=feature_name



import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.01, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(train1[feature_names].fillna(-1), train1['target'])

#print('modeling RMSLE %.5f'% model_lgb.best_score_)
y_pred=model_lgb.predict(test1[feature_names].fillna(-1))

#output
result=pd.DataFrame({'item':test1['item'].unique()})
result['pre']=y_pred
result=result.rename(columns={'item':'Id','pre':'Pred'})
result.to_csv(path_result_out,header=True,index=False)
print('timed used:',(datetime.datetime.now()-start_all).seconds)