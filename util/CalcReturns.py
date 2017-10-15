import math

def CalcReturns(data):
    R = []    
    for i in range(len(data)-1):
    	try:
    		r = math.fabs(data[i+1]/data[i])
        	v = math.log(r)

        	R.append(v)
        except:        	
        	print i, data[i+1], data[i], r, 

    return R