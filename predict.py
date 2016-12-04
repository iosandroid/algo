import csv
import numpy as np
from sklearn.svm import SVR
#import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

#get_data('aapl.csv')
#print(dates)
#print(prices)

def predict_prices():
	global dates
	dates = np.reshape(dates, (len(dates), 1))
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3, degree = 2)
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma = 0.1)

	svr_lin.fit(dates, prices)
	print('here5')
	#svr_poly.fit(dates, prices)
	print('here6')
	svr_rbf.fit(dates,prices)
	return

get_data('aapl.csv')
predict_prices()


