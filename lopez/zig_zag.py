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
    
def zig_zag_df(close, minsize):
    
    tEvents, sEvents, bEvents = [], [], []
    
    Max  = 0
    Min  = 10000000
    
    Flag = False
    
    #PriceLow = Max
    #PriceHigh = Min    
    
    #for i in tqdm(close.index[::-1]):
    for i in tqdm(close.index):

        #PriceLow = close.loc[i]
        #PriceHigh = close.loc[i]

        Price = close.loc[i]

        if Flag:
            
            if Price > Max:
                
                Max = Price                
                Max_i = i

                tEvents.append(Max_i)
                sEvents.append(Max_i)
                
            elif (Max - Price >= minsize):
                
                tEvents.append(Max_i)
                sEvents.append(Max_i)

                Flag = False

                Min = Price
                Min_i = i
                
        else:
               
            if Price < Min:
                    
                Min = Price
                Min_i = i

                bEvents.append(Min_i)
                tEvents.append(Min_i)
                    
            elif (Price - Min >= minsize):
                    
                tEvents.append(Min_i)
                bEvents.append(Min_i)
               
                Flag = True

                Max = Price 
                Max_i = i
        
    return pd.DatetimeIndex(tEvents), pd.DatetimeIndex(sEvents), pd.DatetimeIndex(bEvents)