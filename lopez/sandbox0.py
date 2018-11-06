import sys
import math

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import roc_curve, classification_report

from multiprocessing import cpu_count
from tqdm import tqdm, tqdm_notebook

from mpengine import mpPandasObj
from util import cprintf

from zig_zag import zig_zag_df
from statsmodels.tsa.stattools import adfuller
from statsmodels.distributions.empirical_distribution import ECDF

from entropy_features import plugIn, lempelZiv_lib, konto
from structural_breaks import get_bsadf, get_bsadf0
from sample_weights import mpNumCoEvents, mpSampleW, mpSampleTW, getAvgUniqueness, getIndMatrix
from financial_data_structures import dollar_bar_df
from cross_validation_in_finance import PurgedKFold, cvScore
from labeling import getDailyVol, getTEvents, addVerticalBarrier, getEvents, getBins, getBinsOld, df_returns, df_rolling_autocorr
from fractionally_differentiated_features import fracDiff, fracDiff_FFD, plotMinFFD

def add_aggressor_side_entropy0(features_df, raw_df):
    
    print('Adding aggressor side entropy0...')
    features_df['aggressor_side_entropy0'] = pd.Series(index = features_df.index)
    
    for i in tqdm(range(len(features_df.index)-1)):
        p = raw_df[features_df.index[i]:features_df.index[i+1]].price
        p = p.diff()
        p = (p / np.abs(p)).fillna(0) + 1
    
        msg = ''
        for j in p:
            msg = msg + str(int(j))

        e = konto(msg) 
        features_df.aggressor_side_entropy0[features_df.index[i + 1]] = e['h']
    
    return features_df

def add_aggressor_side_entropy1(features_df, raw_df):

    print('Adding aggressor side entropy1...')
    features_df['aggressor_side_entropy1'] = pd.Series(index = features_df.index)
    
    for i in tqdm(range(len(features_df.index)-1)):
        vol = raw_df[features_df.index[i]:features_df.index[i+1]].volume
        vol = vol / np.abs(vol)
        vol = (vol + 1) / 2
    
        msg = ''
        for j in vol:
            msg = msg + str(int(j))

        e = konto(msg) 
        features_df.aggressor_side_entropy1[features_df.index[i + 1]] = e['h']
    
    return features_df

def add_bsadf(features_df):

    print('Adding bsadf...')
    features_df['bsadf'] = pd.Series(index = features_df.index)
    
    price = features_df.price
    logp = np.log(price)
    
    for i in tqdm(range(1,logp.shape[0])):
        index = features_df.index[i]        
        features_df.bsadf[index] = get_bsadf0(logP = logp[0:i], minSL = 100, constant ='ctt', lags = 16)['gsadf']

    return features_df

def get_quantile_encoding_msg(series, quantile = 10):

    print('Doing quantile encoding msg...')
    Q = np.linspace(.0, 1., quantile + 1)    
    
    msg = '';
    for x in tqdm(series):
        for i in range(len(Q)-1):
          
            q0 = series.quantile(Q[i])
            q1 = series.quantile(Q[i+1])
            
            if x >= q0 and x < q1:
                msg = msg + str(int(i))
                break
                
    return msg

def add_returns_entropy(features_df, q = 2):
    
    print('Adding returns entropy...')
    features_df['returns_entropy'] = pd.Series(index = features_df.index)
    
    s0 = features_df.price[0:-2]
    s1 = features_df.price[1:-1]

    r = np.log(np.divide(s1,s0))
    msg = get_quantile_encoding_msg(r, q)
    
    for i in tqdm(range(100, len(msg))):
        msg0 = msg[0:i]
        e = konto(msg0)
        
        index = features_df.index[i]  
        features_df.returns_entropy[index] = e['h']
      
    return features_df

def add_buys_volume_entropy(features_df, raw_df, q = 2):

    print('Adding buys volume entropy...')
    features_df['buys_volume_entropy'] = pd.Series(index = features_df.index)
    
    s = pd.Series(index = features_df.index)

    for i in tqdm(range(len(features_df.index)-1)):
        df0 = raw_df[features_df.index[i]:features_df.index[i + 1]]

        Vb = df0[df0.volume > 0].volume.sum()
        V = df0.volume.abs().sum()

        index = features_df.index[i+1] 
        s[index] = Vb/V
    
    
    s = s.dropna()
    msg = get_quantile_encoding_msg(s, 2)

    for i in tqdm(range(100, len(msg))):
            msg0 = msg[0:i]
            e = konto(msg0)

            index = features_df.index[i]
            features_df.buys_volume_entropy[index] = e['h']
    
    return features_df

def add_kyles_lambda(features_df, raw_df):    

    print('Adding kyles lambda...')
    features_df['kyles_lambda'] = pd.Series(index = features_df.index)
    
    from sklearn import linear_model
    
    for i in tqdm(range(len(features_df.index)-1)):
        df0 = raw_df[features_df.index[i]:features_df.index[i + 1]]
        
        dP = df0.price.diff()
        dP = dP.dropna()

        bV = df0.volume[dP.index]

        dP = np.array(dP)
        bV = np.array(bV)

        bV = bV.reshape(-1, 1)

        regr = linear_model.LinearRegression()
        regr.fit(bV, dP)
        
        index = features_df.index[i]
        features_df.kyles_lambda[index] = regr.coef_
        
    return features_df

def add_amihuds_lambda(features_df, raw_df, start = 100):

    print('Adding amihuds lambda...')
    features_df['amihuds_lambda'] = pd.Series(index = features_df.index)

    dLogP = np.abs(np.log(features_df.price).diff())
    dLogP = dLogP.dropna()
    
    bV = pd.Series(index = dLogP.index)
    
    from sklearn import linear_model
    
    for i in tqdm(range(len(features_df.index)-1)):
        index0 = features_df.index[i]
        index1 = features_df.index[i + 1]
        
        df0 = raw_df[index0:index1]        
        bV[index1] = df0.volume.sum()
    
    for i in tqdm(range(start,len(dLogP.index))):        
        X = bV.iloc[:i]
        Y = dLogP.iloc[:i]
        
        X = np.array(X)
        Y = np.array(Y)
        
        X = X.reshape(-1, 1)
        
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
    
        index = dLogP.index[i]
        features_df.amihuds_lambda[index] = regr.coef_
        
    return features_df

def add_vpin(features_df, raw_df):

    print('Adding vpin...')
    features_df['vpin'] = pd.Series(index = features_df.index)
    
    Vb = pd.Series(index = features_df.index)
    Vs = pd.Series(index = features_df.index)
    V  = pd.Series(index = features_df.index)
    
    for i in tqdm(range(len(features_df.index)-1)):
        index0 = features_df.index[i]
        index1 = features_df.index[i + 1]
        
        df0 = raw_df[index0:index1]
        
        dfB = df0[df0.volume > 0]
        dfS = df0[df0.volume < 0]
        
        Vb[index1] = np.abs(dfB.volume).sum()
        Vs[index1] = np.abs(dfS.volume).sum()
        
        V[index1] = Vb[index1] + Vs[index1]
        
    V  = V.dropna()
    Vb = Vb.dropna()
    Vs = Vs.dropna()
        
    for i in tqdm(range(1, len(features_df.index))):    
        Vb0 = Vb.iloc[:i]
        Vs0 = Vs.iloc[:i]
        V0  = V.iloc[:i]
        
        index = features_df.index[i]
        features_df.vpin[index] = np.abs(np.subtract(Vb0, Vs0)).sum() / V0.sum()
        
    return features_df   

def main():
	filepath = './data/bitfinex_BTCUSD_trades.csv'
	#filepath = '~/Dev/notebook/lopez/data/btcusd_trades.csv'
	cols = list(map(str.lower, ['Datetime','Amount','Price','<Unknown>']))
	columns = dict(zip(range(len(cols)), cols))

	print('Reading data...')

	df = pd.read_csv(filepath, header = None).rename(columns = columns).assign(dates = lambda df: (pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S.%f'))).assign(dollar_volume=lambda df: df['amount'] * df['price']).assign(volume=lambda df: df['amount']).drop(['datetime', '<unknown>'], axis = 1).set_index('dates').drop_duplicates()
	df = df.iloc[::-1]

	dollar_M = 1000000
	dollar_df = dollar_bar_df(df, 'dollar_volume', dollar_M)
	dollar_df = dollar_df.iloc[0:300]

	close = dollar_df.price.copy()
	close = close[~close.index.duplicated(keep='first')]

	features = pd.DataFrame(index = dollar_df.index, columns = ['price'])
	features.price = close

	features = add_vpin(features, df)
	features.vpin = features.vpin - features.vpin.mean()

	features = add_amihuds_lambda(features, df)
	features.amihuds_lambda = features.amihuds_lambda - features.amihuds_lambda.mean()

	features = add_kyles_lambda(features, df)
	features.kyles_lambda = features.kyles_lambda - features.kyles_lambda.mean()

	features = add_buys_volume_entropy(features, df, q = 4)
	features.buys_volume_entropy = features.buys_volume_entropy - features.buys_volume_entropy.mean()

	features = add_bsadf(features)
	features.bsadf = features.bsadf - features.bsadf.mean()

	features = add_returns_entropy(features)
	features.returns_entropy = features.returns_entropy - features.returns_entropy.mean()

	features = add_aggressor_side_entropy1(features, df)
	features.aggressor_side_entropy1 = features.aggressor_side_entropy1 - features.aggressor_side_entropy1.mean()

	features = features.dropna()
	features.columns

	#f, ax = plt.subplots(3)

	#features.price.plot(ax=ax[0], title = 'Price')
	#features.vpin.plot(ax=ax[1], title = 'vpin')
	#features.amihuds_lambda.plot(ax=ax[2], title = 'amihuds_lambda')
	#features.kyles_lambda.plot(ax=ax[3], title = 'kyles_lambda')
	#features.buys_volume_entropy.plot(ax=ax[4], title = 'buys_volume_entropy')
	#features.bsadf.plot(ax=ax[5], title = 'bsadf')
	#features.returns_entropy.plot(ax=ax[6], title = 'returns_entropy')
	#features.aggressor_side_entropy1.plot(ax=ax[7], title = 'aggressor_side_entropy1')

	features.to_csv('features.csv')
	pass

if __name__ == '__main__':
	main()