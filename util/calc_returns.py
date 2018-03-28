import math
import numpy as np

def LogRet(data):
	R = []
	for i in range(len(data)-1):
		try:
			r = math.fabs(data[i+1]/data[i])
			v = math.log(r)

			R.append(v)
		except:
		 	print i, data[i+1], data[i], r, 

	return np.array(R)

def DifRet(data):
	R = []
	for i in range(len(data)-1):
		try:
			r = (data[i+1]-data[i])/data[i]
			R.append(r)
		except:
			print i, data[i+1], data[i], r, 

	return np.array(R)


def iDifRet(S0, r):
	S = []
	Si = S0
	for ri in r:

		Si = Si*(ri+1)
		S.append(Si)

	return np.array(S)