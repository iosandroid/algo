import math
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')



def ZigZag(price, minsize, plotting = True):
    
    R = []
    N = len(price)

    for i in range(N-1):
        R.append(price[i+1] - price[i])

    N = len(R)

    pref = []
    pref.append({'v' : R[0], 'i' : 1})

    for i in range(1, N):
        pref.append({'v' : pref[i-1]['v'] + R[i], 'i' : i + 1})

    X = []
    N = len(price)
    
    ZigZagInternal(X, pref, minsize, 0, len(pref))

    X.sort(key = lambda tup: tup['tmin'])

    Y = []
    for i in range(len(X)-1):

        X0 = X[i]
        X1 = X[i+1]

        Y.append({'tmin' : X0['tmax'], 'tmax' : X1['tmin'], 'label' : 0, 'slice' : 0})

    if len(X)>0:
        X0 = X[0]
        Xn = X[-1]

        if X0['tmin']>0:
            Y.append({'tmin' : 0, 'tmax' : X0['tmin'], 'label' : 0, 'slice' : 0})

        if Xn['tmax']<N:
            Y.append({'tmin' : Xn['tmax'], 'tmax' : N, 'label' : 0, 'slice' : 0})            


    Z = sorted(X + Y, key = lambda tup : tup['tmin'])


    #### plot ##################################################################################

    if plotting:
        
        fig, ax = plt.subplots()
        for i in range(len(Z)):

            i0    = Z[i]['tmin']
            i1    = Z[i]['tmax']
            label = Z[i]['label']
            slice = Z[i]['slice']

            color = None

            if label < 0:
                color = 'red'
            elif label > 0:
                color = 'blue'
            else:
                color = 'green'

            ax.plot(range(i0, i1), price[i0 : i1], color = color)

            #print i0, i1, i1 - i0, label, slice, price[i0], price[i1], price[i1] - price[i0]        

        plt.show()

    ############################################################################################
    
    B = 0
    S = 0
    H = 0

    for i in range(len(Z)):

        i0 = Z[i]['tmin']
        i1 = Z[i]['tmax']

        label = Z[i]['label']

        if label < 0:
            S = S + (i1-i0)
        elif label > 0:
            B = B + (i1-i0)
        else:
            H = H + (i1-i0)

    print 'Buys: %s; Sells: %s; Holds: %s; Count: %s' % (B,S,H,N)

    ############################################################################################

    return Z

def ZigZagInternal(Z, pref, minsize, pmin, pmax):
    
    Z0    = []
    pref0 = pref[pmin:pmax]

    count = len(pref0)
    

    for i in range(count):
        for p in range(i, count-1):

            slice = pref0[p + 1]['v'] - pref0[i]['v']
            delta = pref0[p + 1]['i'] - pref0[i]['i']

            if math.fabs(slice) > minsize:
                Z0.append({'tmin' : pref0[i]['i'], 'tmax' : pref0[p + 1]['i'], 'delta' : delta, 'slice' : slice, 'label' : 1 if slice > 0 else -1})

                break

    count = len(Z0)

    if count > 0:
        minZ0 = Z0[0]

        for i in range(count):
            delta = Z0[i]['delta']

            if delta < minZ0['delta']:                
                minZ0 = Z0[i]

        Z.append(minZ0)

        zmin = minZ0['tmin'] - 1
        zmax = minZ0['tmax']

        if zmin - pmin > 1:
            ZigZagInternal(Z, pref, minsize, pmin, zmin)

        if pmax - zmax > 1:
            ZigZagInternal(Z, pref, minsize, zmax, pmax)

    pass