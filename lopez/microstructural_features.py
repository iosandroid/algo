import sys
import math

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats

from multiprocessing import cpu_count
from tqdm import tqdm, tqdm_notebook

from mpengine import mpPandasObj

def getBeta(series, sl):
	hl = series[['High', 'Low']].values
	hl = np.log(hl[:, 0]/hl[:, 1]) ** 2
	hl = pd.Series(hl, index = series.index)

	beta = pd.stats.moments.rolling_sum(hl, window = 2)
	beta = pd.stats.moments.rolling_mean(beta, window = sl)

	return beta.dropna()

def getGamma(series):
	h2 = pd.stats.moments.rolling_max(series['High'], window = 2)
	l2 = pd.stats.moments.rolling_min(series['Low'], window = 2)

	gamma = np.log(h2.values / l2.values) ** 2
	gamma = pd.Series(gamma, index= h2.index)

	return gamma.dropna()

def getAlpha(beta, gamma):
	den = 3 - 2*2 ** .5
	alpha  = (2 ** .5 - 1) * (beta ** .5) / den
	alpha -= (gamma / den) ** .5

	alpha[alpha < 0] = 0 # set negative alphas to 0 (see p. 727 of paper)
	return alpha.dropna()

def corwinSchultz(series, sl = 1):
	# Note: S < 0 iif alpha < 0
	beta = getBeta(series, sl)
	gamma = getGamma(series)
	alpha = getAlpha(beta, gamma)
	spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

	startTime = pd.Series(series.index[0 : spread.shape[0]], index = spread.index)

	spread = pd.concat([spread, startTime], axis = 1)
	spread.columns = ['Spread', 'Start_time'] # 1st loc used to compute beta

	return spread