import numpy as np
from util import ZigZag
from sklearn import preprocessing

#def BuildData0(zigzag, data, lag):
    
#    D = []
#    L = []
    
#    D0 = []
#    D1 = []
    
#    L0 = []
#    L1 = []
    
#    N = len(data)
#    Count = len(Z['time'])

#    for i in range(0,N-lag):        
#        try:
#            index = zigzag["time"].index(i+lag)
            
#            D0.append(data[i:i+lag])
#            L0.append(zigzag["label"][index])
            
#        except:
            
#            D1.append(data[i:i+lag])
#            L1.append(0)            

#    Count0 = len(D0)
#    Count1 = len(D1)
    
#    #print(Count0, Count1)
    
#    N = Count0 / 2
#    if Count1 > N:        
#        D1 = np.random.permutation(D1)
        
#        D = D1[:N]
#        L = L1[:N]
        
#    else:        
#        D = D1
#        L = L1
    
#    D_temp = np.concatenate((D, D0), axis=0)
#    L_temp = np.concatenate((L, L0), axis=0)    
    
#    I = range(len(D_temp))
#    I = np.random.permutation(I)
    
#    D = []
#    L = []
    
#    for i in I:
#        D.append(D_temp[i])
#        L.append(L_temp[i])
    
#    D = np.array(D)
#    L = np.array(L)   
    
#    D = preprocessing.scale(D)
    
#    return D, L
    
#def BuildData1(zigzag, returns, lag):
    
#    D = []
#    L = []
    
#    D0 = []
#    D1 = []
    
#    L0 = []
#    L1 = []
    
#    N = len(returns) + 1
#    Count = len(Z['time'])

#    for i in range(0,N-lag):        
#        try:
#            index = zigzag["time"].index(i+lag-1)
            
#            D0.append(returns[i:i+lag-1])
#            L0.append(zigzag["label"][index])
            
#        except:
            
#            D1.append(returns[i:i+lag-1])
#            L1.append(0)            

#    Count0 = len(D0)
#    Count1 = len(D1)
    
#    #print(Count0, Count1)
    
#    N = Count0 / 2
#    #N = Count0
#    if Count1 > N:        
#        D1 = np.random.permutation(D1)
        
#        D = D1[:N]
#        L = L1[:N]
        
#    else:        
#        D = D1
#        L = L1
    
#    D_temp = np.concatenate((D, D0), axis=0)
#    L_temp = np.concatenate((L, L0), axis=0)
    
#    I = range(len(D_temp))
#    I = np.random.permutation(I)
    
#    D = []
#    L = []
    
#    for i in I:
#        D.append(D_temp[i])
#        L.append(L_temp[i])
    
#    D = np.array(D)
#    L = np.array(L)   
    
#    D = preprocessing.scale(D)
    
#    return D, L

def BuildData2(zigzag, returns, lag):
    
    D = []
    L = []
    
    D0 = []
    D1 = []
    
    L0 = []
    L1 = []
    
    N = len(returns) + 1
    Count = len(zigzag['time'])

    for i in range(0,N-lag):        
        try:
            index = zigzag["time"].index(i+lag-1)
            
            D0.append(returns[i:i+lag-1])
            L0.append(zigzag["label"][index])
            
        except:
            
            D1.append(returns[i:i+lag-1])
            L1.append(0)            

    Count0 = len(D0)
    Count1 = len(D1)
    
    #print(Count0, Count1)
    
    N = Count0
    if Count1 > N:        
        D1 = np.random.permutation(D1)
        
        D = D1[:N]
        L = L1[:N]
        
    else:        
        D = D1
        L = L1
    
    D_temp = np.concatenate((D, D0), axis=0)
    L_temp = np.concatenate((L, L0), axis=0)
    
    I = range(len(D_temp))
    I = np.random.permutation(I)
    
    #D = []
    #L = []

    S = {'data' : [], 'label' : [], 'index' : []}
    
    for i in I:
        S['index'].append(i)
        S['data'].append(D_temp[i])
        S['label'].append(L_temp[i])
    
    S['data']  = np.array(S['data'])
    S['label'] = np.array(S['label'])
    S['index'] = np.array(S['index'])
    
    S['data'] = preprocessing.scale(S['data'])

    return S

def BuildData3(zigzag, returns, lag):
    
    D = []
    L = []
    
    D0 = []
    D1 = []
    
    L0 = []
    L1 = []
    
    N = len(returns) + 1
    Count = len(zigzag['time'])

    for i in range(0,N-lag):        
        try:
            index = zigzag["time"].index(i+lag-1)
            
            D0.append(returns[i:i+lag-1])
            
            if zigzag["label"][index] == 1:                
                L0.append([0, 0, 1])
            else:
                L0.append([1, 0, 0])
            
        except:
            
            D1.append(returns[i:i+lag-1])
            L1.append([0, 1, 0])

    Count0 = len(D0)
    Count1 = len(D1)
    
    #print(Count0, Count1)
    
    N = Count0 / 2
    if Count1 > N:        
        D1 = np.random.permutation(D1)
        
        D = D1[:N]
        L = L1[:N]
        
    else:        
        D = D1
        L = L1
    
    D_temp = np.concatenate((D, D0), axis=0)
    L_temp = np.concatenate((L, L0), axis=0)
    
    I = range(len(D_temp))
    I = np.random.permutation(I)
    
    D = []
    L = []
    
    for i in I:
        D.append(D_temp[i])
        L.append(L_temp[i])
    
    D = np.array(D)
    L = np.array(L)   
    
    D = preprocessing.scale(D)
    
    return D, L