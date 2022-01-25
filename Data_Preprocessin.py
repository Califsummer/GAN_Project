#The code in this file is used to preprocess raw data acquired from SCADA system

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
import time
pd.set_option("precision", 20)

df = pd.read_csv('D:\SCADA_Data\RealDataForClassification', float_precision='round_trip')
df = df[(df['Protocol'] == 'Modbus/TCP') | (df['Protocol'] == 'TCP') | (df['Protocol'] == 'ARP')]
df['Function Code'] = df['Function Code'].replace({'Write Multiple Registers' : 16})
df['Function Code'] = df['Function Code'].replace({'Read Holding Registers' : 3})

#Label the class (if attack or not)
def label_race (row):
    if row['Protocol'] == 'ARP':
        return 1
    if row['Fin'] == 'Set' :
        return 1
    if row['Reset'] == 'Set' :
        return 1
    if (row['Reference Number'] == 32210) & ((row['Register Value'] > 9) | (row['Register Value'] < -9)) :
        return 1
    if (row['Reference Number'] == 42210) & ((row['Register Value'] > 95) | (row['Register Value'] < 5)) :
        return 1
    if (row['Reference Number'] == 42211) & ((row['Register Value'] > 95) | (row['Register Value'] < 5)) :
        return 1
    if (row['Reference Number'] == 42212) & (row['Register Value'] != 95) :
        return 1
    if (row['Reference Number'] == 42215) & (row['Register Value'] != 5) :
        return 1
    else :
        return 0
        
df['isAttack'] = df.apply (lambda row: label_race(row), axis=1)

#Map the destination IP address of MITM (man in the middle) to a integer value 
def MItMHash (row):
    if row['Protocol'] == 'ARP':
        return 10000
df['Destination'] = df.apply (lambda row: MItMHash(row), axis=1)

#Map the string variables in some other columns to integer values
df['Fin'] = df['Fin'].replace({'Not set' : 0, 'Set' : 1})
df['Reset'] = df['Reset'].replace({'Not set' : 0, 'Set' : 1})

df['Source'] = df['Source'].replace({'10.0.0.3':1, '10.0.0.4':1000, '10.0.0.5':5000, '10.0.0.111':884431})
df['Destination'] = df['Destination'].replace({'10.0.0.3':1, '10.0.0.4':1000, '10.0.0.5':5000, '10.0.0.111':884431})

df['Protocol'] = df['Protocol'].replace({'TCP':10, 'Modbus/TCP':1000, 'ARP' : 5000})

#Data cleansing
df = df.fillna(0)
df = df.drop(["Syn", "Info"], axis = 1)
df = df.loc[(df["Source"] != 'Schindle_33:ff:38') & (df["Source"] != 'Schindle_cd:b8:04') & (df["Source"] != 'Broadcast')]
df = df.loc[(df["Destination"] != 'Schindle_33:ff:38') & (df["Destination"] != 'Schindle_cd:b8:04') & (df["Destination"] != 'Broadcast')]

df.to_csv('D:\SCADA_Data\ProcessedRealData', index=False)
