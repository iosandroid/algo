import numpy as np

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
    
