import warnings

import numpy as np
import matplotlib.pyplot as plt

from util.ZigZag import ZigZag
from util.CalcReturns import CalcReturns

from scipy.linalg import hankel
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')


def TrainingSet_ML_Prices(prices, minsize, lag, scale = False):

    S = MinMaxScaler()
    P = prices    
    M = minsize

    Z = ZigZag(P, M, True)

    N = len(P)
    T = { 'input' : hankel(P[0 : lag], P[lag-1 :]).T, 'label' : np.full(N-lag+1, 0) }

    #fliter Z according lag
    for i in range(len(Z)):    
        tmin = Z[i]['tmin']
        tmax = Z[i]['tmax']

        if (tmin <= lag) and (lag < tmax):            

            Z[i]['tmin'] = lag
            if i > 0:
                Z = Z[i-1:]

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

def TrainingSet_ML_Logret(prices, minsize, lag, scale = False):   

    P = prices
    N = len(P)

    T0, _ = TrainingSet_ML_Prices(prices, minsize, lag + 1)
    T = { 'input' : np.full((N-lag, lag), 0), 'label' : T0['label'] }

    for i in range(len(T0['input'])):
        T['input'][i] = CalcReturns(T0['input'][i])

    S = MinMaxScaler()
    if scale:
        P = S.fit_transform(CalcReturns(P))

        for i in range(len(T)):
            T['input'][i] = S.transform(T['input'][i])

    return T, S


def TrainingSet_NN_Prices(prices, minsize, lag, scale = False):

    T, S = TrainingSet_ML_Prices(prices, minsize, lag, scale)
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

def TrainingSet_NN_Logret(prices, minsize, lag, scale = False):

    T, S = TrainingSet_ML_Logret(prices, minsize, lag, scale)
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

# Build data without trend removing
#def BuildData0(zigzag, prices):
#    
#    scaler = MinMaxScaler()

#    D = []
#    L = []

#    for i in range(len(zigzag["time"])-1):
#        index0 = zigzag["time"][i + 0]
#        index1 = zigzag["time"][i + 1]

#        label  = zigzag["label"][i + 0]
        
#        for j in range(index0, index1):
#            D.append(prices[j])
#            L.append([0, 1] if label == 1 else [1, 0])

#    N = len(D)

#    D = np.array(D)
#    L = np.array(L)

#    D = scaler.fit_transform(D)

#    #### plot #############################################
#    fig, ax = plt.subplots()
#    for i in range(len(zigzag["time"])-1):
#        i0 = zigzag["time"][i + 0]
#        i1 = zigzag["time"][i + 1]

#        lb = zigzag["label"][i + 0]

#        ax.plot(range(i0,i1), D[i0:i1], color = 'green' if lb == 1 else 'red')

#    plt.show()
#    ######################################################


#    #D = D.reshape(N,1,1)

#    T = {'input' : [], 'label' : []}
    
#    T['input'] = D
#    T['label'] = L    

#    return T, scaler

#def BuildData2(zigzag, returns, lag):
#    
#    D = []
#    L = []
#    
#    D0 = []
#    D1 = []
#    
#    L0 = []
#    L1 = []
    
#    N = len(returns) + 1
#    Count = len(zigzag['time'])

#    for i in range(0,N-lag):        
#        try:
#            index = zigzag["time"].index(i+lag-1)
#            
#            D0.append(returns[i:i+lag-1])
#            L0.append(zigzag["label"][index])
#            
#        except:
#            
#            D1.append(returns[i:i+lag-1])
#            L1.append(0)            
#
#    Count0 = len(D0)
#    Count1 = len(D1)
#    
#    #print(Count0, Count1)
#    
#    N = Count0
#    if Count1 > N:        
#        D1 = np.random.permutation(D1)
#        
#        D = D1[:N]
#        L = L1[:N]
#        
#    else:        
#        D = D1
#        L = L1
#    
#    D_temp = np.concatenate((D, D0), axis=0)
#    L_temp = np.concatenate((L, L0), axis=0)
#    
#    I = range(len(D_temp))
#    I = np.random.permutation(I)
#    
#    #D = []
#    #L = []
#
#    S = {'data' : [], 'label' : [], 'index' : []}
#    
#    for i in I:
#        S['index'].append(i)
#        S['data'].append(D_temp[i])
#        S['label'].append(L_temp[i])
#    
#    S['data']  = np.array(S['data'])
#    S['label'] = np.array(S['label'])
#    S['index'] = np.array(S['index'])
#    
#    S['data'] = preprocessing.scale(S['data'])
#
#    return S

#def BuildData3(zigzag, returns, lag):
#    
#    D = []
#    L = []
#    
#    D0 = []
#    D1 = []
#    
#    L0 = []
#    L1 = []
#    
#    N = len(returns) + 1
#    Count = len(zigzag['time'])
#
#    for i in range(0,N-lag):        
#        try:
#            index = zigzag["time"].index(i+lag-1)
#            
#            D0.append(returns[i:i+lag-1])
#            
#            if zigzag["label"][index] == 1:                
#                L0.append([0, 0, 1])
#            else:
#                L0.append([1, 0, 0])
#            
#        except:
#            
#            D1.append(returns[i:i+lag-1])
#            L1.append([0, 1, 0])
#
#    Count0 = len(D0)
#    Count1 = len(D1)
#    
#    #print(Count0, Count1)
#    
#    N = Count0 / 2
#    if Count1 > N:        
#        D1 = np.random.permutation(D1)
#        
#        D = D1[:N]
#        L = L1[:N]
#        
#    else:        
#        D = D1
#        L = L1
#    
#    D_temp = np.concatenate((D, D0), axis=0)
#    L_temp = np.concatenate((L, L0), axis=0)
#    
#    I = range(len(D_temp))
#    I = np.random.permutation(I)
#   
#    D = []
#    L = []
#    
#    for i in I:
#        D.append(D_temp[i])
#        L.append(L_temp[i])
#     
#    D = np.array(D)
#    L = np.array(L)   
#    
#    D = preprocessing.scale(D)
#    
#    return D, L

#def BuildData4(zigzag, returns, lag):
#    
#    D = []
#    L = []
#  
#    N = len(returns) + 1
#    Count = len(zigzag['time'])

#    #returns = preprocessing.scale(returns)
#
#    for i in range(len(zigzag["time"])-1):
#        index0 = zigzag["time"][i + 0]
#        index1 = zigzag["time"][i + 1]
#
#        l = zigzag["label"][i + 0]
        
#        for j in range(index0, index1):
#            r = returns[j-lag+1:j]            
#            if len(r) is not 0:
#                D.append(r)
#                L.append(l)
#            
#
#    S = {'data' : [], 'label' : []}
#    
#    S['data']  = np.array(D)
#    S['label'] = np.array(L)    
#
#    return S