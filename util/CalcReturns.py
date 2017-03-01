import math

def CalcReturns(data):
    R = []    
    for i in range(len(data)-1):
        v = math.log(math.fabs(data[i+1]/data[i]))
        R.append(v)

    return R