# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:46:33 2018

@author: Wizza
"""

import pandas as pd

train_log_address=r'D:\LJK\credit plan\train_log.csv'
train_agg_address=r'D:\LJK\credit plan\train_agg.csv'

train_log=pd.read_csv(train_log_address)
train_log.head()
train_log=train_log['USRID\tEVT_LBL\tOCC_TIM\tTCH_TYP'].str.split('\t',expand=True)
train_log.columns=['USRID\tEVT_LBL\tOCC_TIM\tTCH_TYP'][0].split('\t')

train_agg=pd.read_csv(train_agg_address)
train_agg_column=train_agg.columns.values
train_agg=train_agg[train_agg_column[0]].str.split('\t',expand=True)
train_agg.columns=train_agg_column[0].split('\t')
train_agg['USRID']=train_agg['USRID'].astype('int')
train_agg=train_agg.sort_values(by='USRID')
train_log_grouped=train_log.groupby(['USRID','EVT_LBL']).count()
