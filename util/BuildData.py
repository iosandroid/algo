import warnings

import numpy as np
import matplotlib.pyplot as plt

from util.ZigZag import ZigZag
from util.CalcReturns import CalcReturns

from scipy.linalg import hankel
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def TrainingSet_ML_Prices(prices, minsize, lag, scale = False, ZigZagFunc = ZigZag):

    S = MinMaxScaler()
    P = prices    
    M = minsize

    Z = ZigZagFunc(P, M, True)

    N = len(P)
    T = { 'input' : hankel(P[0 : lag], P[lag-1 :]).T, 'label' : np.full(N-lag+1, 0) }

    #fliter Z according lag
    for i in range(len(Z)):    
        tmin = Z[i]['tmin']
        tmax = Z[i]['tmax']

        if (tmin <= lag) and (lag < tmax):            

            Z[i]['tmin'] = lag
            Z = Z[i:]

            break

    for i in range(len(Z)):
        tmin = Z[i]['tmin']
        tmax = Z[i]['tmax']

        T['label'][tmin-lag : tmax-lag] = np.full(tmax - tmin, Z[i]['label'])

    S = MinMaxScaler()
    if scale:
        P = S.fit_transform(P)

        for i in range(len(T)):
            T['input'][i] = S.transform(T['input'][i])

    return T, S

def TrainingSet_ML_Logret(prices, minsize, lag, scale = False, ZigZagFunc = ZigZag):

    P = prices
    N = len(P)

    T0, _ = TrainingSet_ML_Prices(prices = prices, minsize = minsize, lag = lag + 1, scale = False, ZigZagFunc = ZigZagFunc)
    T = { 'input' : np.full((N-lag, lag), 0), 'label' : T0['label'] }

    for i in range(len(T0['input'])):
        T['input'][i] = CalcReturns(T0['input'][i])

    S = MinMaxScaler()
    if scale:
        P = S.fit_transform(CalcReturns(P))

        for i in range(len(T)):
            T['input'][i] = S.transform(T['input'][i])

    return T, S


def TrainingSet_NN_Prices(prices, minsize, lag, scale = False, ZigZagFunc = ZigZag):

    T, S = TrainingSet_ML_Prices(prices, minsize, lag, scale, ZigZagFunc)
    N = len(T['label'])

    L = np.full((N, 3), 0)

    for i in range(N):
        label = T['label'][i]

        if label > 0:
            L[i] = np.array([0, 0, 1])
        elif label < 0:
            L[i] = np.array([1, 0, 0])
        else:
            L[i] = np.array([0, 1, 0])

    T['label'] = L

    return T, S

def TrainingSet_NN_Logret(prices, minsize, lag, scale = False, ZigZagFunc = ZigZag):

    T, S = TrainingSet_ML_Logret(prices, minsize, lag, scale, ZigZagFunc)
    N = len(T['label'])

    L = np.full((N, 3), 0)

    for i in range(N):
        label = T['label'][i]

        if label > 0:
            L[i] = np.array([0, 0, 1])
        elif label < 0:
            L[i] = np.array([1, 0, 0])
        else:
            L[i] = np.array([0, 1, 0])

    T['label'] = L

    return T, S

