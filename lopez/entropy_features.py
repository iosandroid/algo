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

def plugIn(msg, w):
	# Compute plug-in (ML) entropy rate
	pmf = pmf1(msg, w)
	out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / w

	return out, pmf

def pmf1(msg, w):
	# Compute the prob mass function for a one-dim discrete rv
	# len(msg) - w occurrences

	try:
		xrange
	except NameError:
		xrange = range
	
	lib = {}
	if not isinstance(msg, str):
		msg = ''.join(map(str, msg))

	for i in xrange(w, len(msg)):
		msg_ = msg[i - w : i]
		if msg_ not in lib:
			lib[msg_] = [i-w]
		else:
			lib[msg_] = lib[msg_] + [i-w]

	pmf = float(len(msg) - w)
	pmf = { i : len(lib[i])/ pmf for i in lib }

	return pmf

def lempelZiv_lib(msg):

	try:
		xrange
	except NameError:
		xrange = range

	i, lib = 1, [msg[0]]
	while i < len(msg):

		for j in xrange(i, len(msg)):
			msg_ = msg[i:j+1]
			if msg_ not in lib:
				lib.append(msg_)
				break

		i = j + 1

	return lib

def matchLength(msg, i, n):
	# Maximum matched length + 1, with overlap.
	# i >= n & len(msg) >= i+n

	try:
		xrange
	except NameError:
		xrange = range

	subS = ''
	for l in xrange(n):
		msg1 = msg[i : i + l + 1]
		for j in xrange(i - n, i):
			msg0 = msg[j : j + l + 1]
			if msg1 == msg0:
				subS = msg1
				break # search for higher l.

	return len(subS) + 1, subS #matched length + 1

def konto(msg, window = None):
	'''
	* Kontoyiannis' LZ entropy estimate, 2013 version (centered window).
	* Inverse of the avg length of the shortest non-redundant substring.
	* if non-redundant substrings are short, the test is highly entropic.
	* window == None for expanding window, in which case len(msg) % 2 == 0
	* If the end of msg is more relevant, try konto(msg[::-1])
	'''

	try:
		xrange
	except NameError:
		xrange = range

	out = {'num' : 0, 'sum' : 0, 'subS' : []}
	if not isinstance(msg, str):
		msg = ''.join(map(str, msg))

	if window is None:
		points = xrange(1, int(len(msg) / 2) + 1)
	else:
		window = min(window, len(msg) / 2)
		points = xrange(window, len(msg) - window + 1)

	for i in points:
		if window is None:
			l, msg_ = matchLength(msg, i, i)
			out['sum'] += np.log2(i + 1) / l # to avoid Doeblin condition
		else:
			l, msg_ = matchLength(msg, i, window)
			out['sum'] += np.log2(window + 1) / l # to avoid Doelbin condition

		out['subS'].append(msg_)
		out['num'] += 1

	out['h'] = out['sum'] / out['num']
	out['r'] = 1 - out['h'] / np.log2(len(msg)) #redundancy, 0 <= r <= 1

	return out
