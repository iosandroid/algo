import sys
import math

import time
import numpy as np
import pandas as pd

from tqdm import tqdm, tqdm_notebook

def zig_zag(data, minsize):
    
    N = len(data) - 1
    Z = {'zigzag': [], 'time': []}
    
    Count = 0;
    
    Max  = 0
    Min  = 0
    
    Flag = False
    
    PriceLow = 0
    PriceHigh = 0    
        
    while N >= 0:
        PriceLow = data[N]
        PriceHigh = data[N]        
        
        if Flag:
            
            if PriceHigh > Max:
                
                Max = PriceHigh                
                
            elif (Max - PriceLow >= minsize):
                
                Z['time'].append(N)
                Z['zigzag'].append(Max)
                
                Count = Count + 1
                
                Flag = False
                Min = PriceLow
                
        else:
               
            if PriceLow < Min:
                    
                Min = PriceLow
                    
            elif (PriceHigh - Min >= minsize):
                    
                Z['time'].append(N)
                Z['zigzag'].append(Min)
                
                Count = Count + 1
                
                Flag = True
                Max = PriceHigh    
    
        N = N - 1
        
    Z['time'] = np.array(list(reversed(Z['time'])))
    Z['zigzag'] = np.array(list(reversed(Z['zigzag'])))
    
    return Z
    
def zig_zag_df0(close, minsize):
    
    tEvents, sEvents, bEvents = [], [], []
    
    Max  = 0
    Min  = 0
    
    Flag = False
    
    PriceLow = 0
    PriceHigh = 0    
    
    print('here0')

    for i in tqdm(close.index[::-1]):

        PriceLow = close.loc[i]
        PriceHigh = close.loc[i]
        
        if Flag:
            
            if PriceHigh > Max:
                
                Max = PriceHigh                
                
            elif (Max - PriceLow >= minsize):
                
                tEvents.append(i)
                sEvents.append(i)

                Flag = False
                Min = PriceLow
                
        else:
               
            if PriceLow < Min:
                    
                Min = PriceLow
                    
            elif (PriceHigh - Min >= minsize):
                    
                tEvents.append(i)
                bEvents.append(i)
               
                Flag = True
                Max = PriceHigh    
        
    print('here')
    return pd.DatetimeIndex(tEvents), pd.DatetimeIndex(sEvents), pd.DatetimeIndex(bEvents)