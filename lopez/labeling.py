import sys
import math

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats

from multiprocessing import cpu_count
from tqdm import tqdm, tqdm_notebook

from mpengine import mpPandasObj

# Daily Volatility Estimator [3.1]
def getDailyVol(close,span0=100):
    df0 = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0 = df0[df0>0]
    df0 = pd.Series(close.index[df0-1], index = close.index[close.shape[0]-df0.shape[0]:])

    try:
        df0 = close.loc[df0.index] / close.loc[df0.values].values - 1
    except Exception as e:
        print ('error: {e}\n please confirm duplicate indices')

    df0 = df0.ewm(span=span0).std().rename('dailyVol')
    return df0

# Symmetric CUSUM Filter [2.5.2.1]
def getTEvents(gRaw, h):
    tEvents, sPos, sNeg = [], 0, 0
    diff = np.log(gRaw).diff().dropna().abs()

    for i in tqdm(diff.index[1:]):
        try:
            pos, neg = float(sPos + diff.loc[i]), float(sNeg + diff.loc[i])
        except Exception as e:
            print(e)
            print(sPos + diff.loc[i], type(sPos + diff.loc[i]))
            print(sNeg + diff.loc[i], type(sNeg + diff.loc[i]))
            break
        
        sPos, sNeg = max(0.,pos), min(0.,neg)
        if sNeg <- h:
            sNeg = 0
            tEvents.append(i)
        
        if sPos > h:
            sPos = 0
            tEvents.append(i)

    return pd.DatetimeIndex(tEvents)

# Adding Vertical Barrier [3.4]
def addVerticalBarrier(tEvents, close, numDays=1):
    t1 = close.index.searchsorted(tEvents + pd.Timedelta(days = numDays))
    t1 = t1[t1 < close.shape[0]]
    t1 = (pd.Series(close.index[t1], index = tEvents[:t1.shape[0]]))
    return t1

# Triple-Barrier Labeling Method [3.2]
def applyPtSlOnT1(close,events,ptSl,molecule):
    # apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    if ptSl[0]>0: 
        pt = ptSl[0]*events_['trgt']
    else: 
        pt = pd.Series(index=events.index) # NaNs
        
    if ptSl[1]>0: 
        sl =- ptSl[1]*events_['trgt']
    else: 
        sl = pd.Series(index=events.index) # NaNs
        
    for loc,t1 in events_['t1'].fillna(close.index[-1]).iteritems():
        df0 = close[loc:t1] # path prices
        df0 = (df0/close[loc]-1) * events_.at[loc,'side'] # path returns

        out.loc[loc,'sl'] = df0[df0 < sl[loc]].index.min() # earliest stop loss
        out.loc[loc,'pt'] = df0[df0 > pt[loc]].index.min() # earliest profit taking
    return out

# Gettting Time of First Touch (getEvents) [3.3], [3.6]
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
    #1) get target
    trgt = trgt.loc[tEvents]
    trgt = trgt[trgt > minRet] # minRet
    
    #2) get t1 (max holding period)
    if t1 is False:
    	t1 = pd.Series(pd.NaT, index = tEvents)
        
    #3) form events object, apply stop loss on t1
    if side is None:
        side_,ptSl_ = pd.Series(1.,index = trgt.index), [ptSl[0],ptSl[0]]
    else: 
        side_, ptSl_ = side.loc[trgt.index],ptSl[:2]
        
    events = (pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis = 1).dropna(subset = ['trgt']))
    
    df0 = mpPandasObj(func = applyPtSlOnT1, pdObj = ('molecule',events.index),
                    numThreads = numThreads, close = close, events = events,
                    ptSl = ptSl_)
    
    events['t1'] = df0.dropna(how = 'all').min(axis = 1) # pd.min ignores nan
    
    if side is None:
        events = events.drop('side',axis = 1)
        
    return events

# Labeling for side and size [3.5]
def getBinsOld(events,close):
    #1) prices aligned with events
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px,method='bfill')

    #2) create out object
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    out['bin'] = np.sign(out['ret'])
    # where out index and t1 (vertical barrier) intersect label 0
    try:
        locs = out.query('index in @t1').index
        out.loc[locs, 'bin'] = 0
    except:
        pass
    return out

# Expanding getBins to Incorporate Meta-Labeling [3.7]
def getBins(events, close):
    '''
    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:
    -events.index is event's starttime
    -events['t1'] is event's endtime
    -events['trgt'] is event's target
    -events['side'] (optional) implies the algo's position side
    Case 1: ('side' not in events): bin in (-1,1) <-label by price action
    Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    '''
    #1) prices aligned with events
    events_ = events.dropna(subset = ['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px,method = 'bfill')

    #2) create out object
    out = pd.DataFrame(index = events_.index)
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1

    if 'side' in events_:
    	out['ret']*=events_['side'] # meta-labeling

    out['bin'] = np.sign(out['ret'])

    if 'side' in events_:
    	out.loc[out['ret'] <= 0,'bin'] = 0 # meta-labeling

    return out

# Dropping Unnecessary Labels [3.8]
def dropLabels(events, minPct=.05):
    # apply weights, drop labels with insufficient examples
    while True:
        df0=events['bin'].value_counts(normalize=True)
        if df0.min()>minPct or df0.shape[0]<3:break
        print('dropped label: ', df0.argmin(),df0.min())
        events=events[events['bin']!=df0.argmin()]
    return events