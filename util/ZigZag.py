def ZigZag(data, minsize):
    
    N = 0
    T = 0
    Z = {'zigzag': [], 'time': [], 'label' : []}    
    
    Max  = data[0]
    Min  = data[0]
    
    Flag = True
    
    PriceLow = 0
    PriceHigh = 0    
        
    while N < len(data):
        
        PriceLow = data[N]
        PriceHigh = data[N]       
        

        if Flag:
            
            if PriceHigh > Max:             
             
             	T = N
                Max = PriceHigh                
                
            elif (Max - PriceLow >= minsize):
                
                Z['time'].append(T)
                Z['label'].append(-1)
                Z['zigzag'].append(Max)
                
                T = N
                Flag = False
                Min = PriceLow                
                
        else:
               
            if PriceLow < Min:
                
                T = N
                Min = PriceLow                
                    
            elif (PriceHigh - Min >= minsize):
                    
                Z['time'].append(T)
                Z['label'].append(1)
                Z['zigzag'].append(Min)
                
                T = N
                Flag = True
                Max = PriceHigh    
    
        N = N + 1    

    return Z