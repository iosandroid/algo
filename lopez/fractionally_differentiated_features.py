import sys
import math

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import missingno as msno
import seaborn as sns
import scipy.stats as stats

from mpengine import mpPandasObj
from tqdm import tqdm, tqdm_notebook
from multiprocessing import cpu_count

def getWeights(d, size):
	w = [1.]
	for k in range(1, size):
		w_ = -w[-1] / k * (d -k + 1)
		w.append(w_)

	w = np.array(w[::-1]).reshape(-1, 1)
	return w

def getWeights_FFD(d, thres):
	# thres > 0 drops insignificant weights

	k = 1
	w = [1.]
	while np.fabs(w[-1]) > thres:
		w_ = -w[-1] / k * (d - k + 1)
		w.append(w_)

		k = k + 1

	w = np.array(w[::-1]).reshape(-1, 1)
	return w

def plotWeights(dRange, nPlots, size):
	w = pd.DataFrame()
	for d in np.linspace(dRange[0], dRange[1], nPlots):
		w_ = getWeights(d, size = size)
		w_ = pd.DataFrame(w_, index = range(w_.shape[0])[::-1], columns = [d])
		w = w.join(w_, how = 'outer')

	ax = w.plot()
	ax.legend(loc = 'upper left')
	mpl.show()

	return

def fracDiff(series, d, thres = 0.01):
	'''
	Increasing width window, with treatment on NaNs
	Note 1: For thres = 1, nothing is skipped.
	Note 2: d can be any positive fractional, not necessarily bounded [0, 1].
	'''

	#1) Compute weight for the longest series
	w = getWeights(d, series.shape[0])

	#2) Determive initial calcs to be skipped based on weight-loss threshold
	w_ = np.cumsum(abs(w))
	w_ /= w_[-1]

	skip = w_[w_ > thres].shape[0]

	#3) Apply weights to values
	df = {}
	for name in series.columns:		
		seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()

		for iloc in range(skip, seriesF.shape[0]):
			loc = seriesF.index[iloc]
			if not np.isfinite(series.loc[loc, name]):
				continue # exclude NAs

			df_[loc] = np.dot(w[-(iloc+1):, :].T, seriesF.loc[:loc])[0,0]

		df[name] = df_.copy(deep = True)
	df = pd.concat(df, axis = 1)
	return df

def fracDiff_FFD(series, d, thres = 1e-4):
	'''
	Constant width window (new solution)
	Note 1: thres determines the cut-off weight for the window
	Node 2: d can be any positive fractional, not necessatily bounded [0, 1]
	'''

	#1) Compute weights for the longest series
	w = getWeights_FFD(d, thres)
	width = len(w) - 1

	print ('width: %d; d: %f' % (width, d))

	#2) Apply weights to values
	df = {}

	for name in series.columns:
		seriesF, df_ = series[[name]].fillna(method='ffill').dropna(), pd.Series()
		for iloc1 in range(width, seriesF.shape[0]):
			loc0, loc1 = seriesF.index[iloc1 - width], seriesF.index[iloc1]
			if not np.isfinite(series.loc[loc1, name]):
				continue # exclude NAs

			df_[loc1] = np.dot(w.T, seriesF.loc[loc0:loc1])[0,0]

		df[name] = df_.copy(deep = True)
	df = pd.concat(df, axis = 1)
	return df

def plotMinFFD(df0):
	from statsmodels.tsa.stattools import adfuller

	path, instName = './', 'ES1_Index_Method12'

	out = pd.DataFrame(columns = ['adfStat','pVal','lags','nObs', '95% conf', 'corr'])
	#df0 = pd.read_csv(path + instName + '.csv', index_col = 0, parse_dates = True)

	for d in np.linspace(0, 1, 11):
		df1 = np.log(df0[['price']])#.resample('1D').last() #downcast to daily obs
		df2 = fracDiff_FFD(df1, d)

		corr = np.corrcoef(df1.loc[df2.index, 'price'], df2['price'])[0,1]

		df2 = adfuller(df2['price'], maxlag = 1, regression = 'c', autolag = None)
		out.loc[d] = list(df2[:4]) + [df2[4]['5%']] + [corr] # with critical value

	out.to_csv(path + instName + '_testMinFFD.csv')
	out[['adfStat','corr']].plot(secondary_y = 'adfState')

	mpl.axhline(out['95% conf'].mean(), linewidth = 1, color = 'r', linestyle = 'dotted')
	mpl.axhline(0, linewidth = 1, color = 'g', linestyle = 'dotted')
	mpl.savefig(path + instName + '_testMinFFD.png')
	
	return out