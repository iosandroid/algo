import sys
import math

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats

from mpengine import mpPandasObj
from tqdm import tqdm, tqdm_notebook
from multiprocessing import cpu_count

# Estimating the uniqueness of a label
def mpNumCoEvents(closeIdx, t1, molecule):
    '''
    Compute the number of concurrent events per bar
    +molecule[0] is the date of the first event on which the weight be computed
    +molecule[-1] is the date of the last event on which the weight be computed
    Any event that start before t1[molecule].max() impacts the count.
    '''
    
    #1) find events that span the period [molecule[0], molecule[-1]]
    
    t1 = t1.fillna(closeIdx[-1]) #unclosed events still must impact other weight
    t1 = t1[t1 >= molecule[0]] #events that end at or after molecule[0]
    t1 = t1.loc[:t1[molecule].max()] #events that start at or before t1[molecule]
    
    #2) count events spanning a bar
    iloc = closeIdx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index = closeIdx[iloc[0]:iloc[1] + 1])
    
    for tIn, tOut in t1.iteritems():
        count.loc[tIn:tOut] += 1
        
    return count.loc[molecule[0]:t1[molecule].max()]

# Estimating the average uniqueness of a label
def mpSampleTW(t1, numCoEvents, molecule):
    #Derive average uniqueness over the event's lifespan
    wght = pd.Series(index = molecule)
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (1./numCoEvents.loc[tIn:tOut]).mean()
        
    return wght

# Build an indicator matrix
def getIndMatrix(barIx, t1):    
    #Get indicator matrix
    
    indM = pd.DataFrame(0, index = barIx, columns = range(t1.shape[0]))
    for i, (t0,t1) in enumerate(t1.iteritems()):
        indM.loc[t0:t1, i] = 1
        
    return indM

# Compute average uniqueness
def getAvgUniqueness(indM):
    #Average uniqueness from indicator matrix
    
    c = indM.sum(axis = 1) #concurrency
    u = indM.div(c, axis = 0) #uniqueness
    avgU = u[u > 0].mean() # average uniqueness
    return avgU


# Return sample from sequential bootstrap
def seqBootstrap(indM, sLength = None):
    #Generate a sample via sequential bootstrap
    if sLength is None:
        sLength = indM.shape[1]
    phi = []
    while len(phi) < sLength:
        avgU = pd.Series()
        for i in indM:
            indM_ = indM[phi + [i]] #reduce indM
            avgU.loc[i] = getAvgUniqueness(indM_).iloc[-1]
        prob = avgU / avgU.sum() #draw prob
        phi += [np.random.choice(indM.columns, p = prob)]
        
    return phi

# Generating a random t1 series
def getRndT1(numObs, numBars, maxH):
    #random t1 Series
    
    t1 = pd.Series()
    for i in xrange(numObs):
        ix = np.random.randint(0, numBars)
        var = ix + np.random.randint(1, maxH)
        t1.loc[ix] = val
        
    return t1.sort_index()

# Uniqueness from standart and sequential bootstraps
def auxMC(numObs, numBars, maxH):
    #Parallelized auxiliary function
    
    t1 = getRndT1(numObs, numBars, maxH)
    barIx = range(t1,max() + 1)
    indM = getIndMAtrix(barIx, t1)
    phi = np.random.choice(indM.columns, size = ind.shape[1])
    stdU = getAvgUniqueness(indM[phi].mean())
    phi = seqBootstrap(indM)
    seqU = getAvgUniqueness(indM[phi]).mean()
    return { 'stdU' : stdU, 'seqU' : seqU }

# Determination of sample weight by absolute return attribution
def mpSampleW(t1, numCoEvents, close, molecule):
    #Derive sample wight by return attribution
    
    ret = np.log(close).diff() # log-returns, so that they are additive
    wght = pd.Series(index = molecule)
    
    for tIn, tOut in t1.loc[wght.index].iteritems():
        wght.loc[tIn] = (ret.loc[tIn:tOut] / numCoEvents.loc[tIn : tOut]).sum()
    return wght.abs()

#out['w'] = mpPandasObj(mpSampleW, ('molecule', events.index), numThreads, t1 = events['t1'], numCoEvents = numCoEvents, close = close)
#out['w'] *= out.shape[0] / out['w'].sum()

# Implementation of time decay factors
def getTimeDecay(tW, clfLastW = 1.):
    #apply piecewise-linear decay to observed uniqueness (tW)
    #newest observation gets weight = 1, oldest observation gets weight = clfLastW
    
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1. - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1. / ((clfLastW + 1) * clfW.iLoc[-1])
        
    const = 1. - slope * clfW.iloc[-1]
    clfW = const + slope * clfW
    clfW[clfW < 0] = 0
    print const, clope
    return clfW