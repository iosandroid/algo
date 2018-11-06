import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as mpl

from mpengine import mpPandasObj

def getSignal(events, stepSize, prob, pred, numClasses, numThreads, **kargs):

	# get signal from predictions

	if prob.shape[0] == 0:
		return pd.Series()

	# 1) generate signals from multinominal classification (one-vs-rest, OvR)

	signal0 = (prob - 1. / numClasses) / (prob * (1. - prob))**.5 # t-value of OvR
	signal0 = pred * (2 * norm.cdf(signal0) - 1) # signal = side * size

	if 'side' in events:
		signal0 *= events.loc[signal0.index, 'side'] # meta-labeling

	# 2) compute average signal among those concurrently open

	df0 = signal0.to_frame('signal').join(events[['t1']], how = 'left')
	df0 = avgActiveSignals(df0, numThreads)

	signal1 = discreteSignal(signal0 = df0, stepSize = stepSize)
	return signal1

def avgActiveSignals(signals, numThreads):
	# compute the average signal among those active

	# 1) time points where signals change (either one starts or one ends)
	tPnts = set(signals['t1'].dropna().values)
	tPnts = tPnts.union(signals.index.values)
	tPnts = list(tPnts)
	tPnts.sort()

	out = mpPandasObj(mpAvgActiveSignals, ('molecule', tPnts), numThreads, signals = signals)
	return out

def mpAvgActiveSignals(signals, molecule):
	'''
	At time loc, average signal among those still active.
	Signal is active if:
		a) issued before or at loc AND
		b) loc before signal's endtime, or endtime is still unknown (NaT)
	'''

	out = pd.Series()
	for loc in molecule:

		df0 = (signals.index.values <= loc) & ((loc < signals['t1']) | pd.isnull(signals['t1']))
		act = signals[df0].index

		if len(act) > 0:
			out[loc] = signals.loc[act, 'signals'].mean()
		else:
			out[loc] = 0 # no signals active at this time

	return out

def discreteSignal(signal0, stepSize):
	# discretize signal

	signal1 = (signal0 / stepSize).round() * stepSize # discretize
	signal1[signal1 > 1] = 1 # cap
	signal1[signal1 < -1] = -1 # floor

	return signal1

def betSize(w, x):
	return x * (w + x**2)**-.5

def getTPos(w, f, mP, maxPos):
	return int(betSize(w, f - mP) * maxPos)

def invPrice(f, w, m):
	return f - m*(w / (1 - m**2)) ** .5

def limitPrice(tPos, pos, f, w, maxPos):
	sgn = (1 if tPos >= pos else -1)
	lP = 0
	for j in xrange(abs(pos + sgn), abs(tPos + 1)):
		lP += invPrice(f, w, j / float(maxPos))

	lP /= tPos - pos
	return lP

def getW(x, m):
	# 0 < alpha < 1
	return (x**2) * ((m**-2) - 1)

pos, maxPos, mP, f, wParams = 0, 100, 100, 115, {'divergence' : 10, 'm' : .95}
w = getW(wParams['divergence'], wParams['m']) # calibrate w
tPos = getTPos(w,f, mP, maxPos) # get tPos
lP = limitPrice(tPos, pos, f, w, maxPos) # limit price for order

