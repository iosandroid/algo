import sys
import csv
import gdax
import time
import datetime

from datetime import timedelta
from dateutil import parser

def main():

	if len(sys.argv) < 2:
		print 'Usage: python pull_gdax.py <symbol>'
		return

	symbol = sys.argv[1]
	
	public_client = gdax.PublicClient()
	delta = datetime.timedelta(seconds = 60 * 300)

	time_data = []

	time0 = datetime.datetime.now()
	for i in range(100):
		time1 = time0
		time0 = time1 - delta

		time_data = [[time0, time1]] + time_data

	csv_data = []

	i = 0
	while i < len(time_data):
		timedata = time_data[i]

		historic_rates = public_client.get_product_historic_rates(symbol, start = timedata[0], end = timedata[1], granularity = 60)
		if type(historic_rates) is dict:
			time.sleep(1)
			continue

		historic_rates.sort(key = lambda r: r[0])

		for row in historic_rates:

			s = datetime.datetime.fromtimestamp(row[0])
			new_row = {
				'Date'   : s.date().strftime('%Y.%m.%d'), 
				'Time'   : s.time().strftime('%H:%M'), 
				'Open'   : row[1], 
				'High'   : row[2], 
				'Low'    : row[3], 
				'Close'  : row[4], 
				'Volume' : row[5]
			}

			csv_data.append(new_row)

		i = i + 1

	lhs, rhs = symbol.split('-')
	filename = 'gdax_' + lhs + rhs + '1.csv'

	#write csv
	with open(filename, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(['Date','Time','Open', 'High', 'Low', 'Close', 'Volume'])

		for elem in csv_data:

			row = [elem['Date'],elem['Time'],elem['Open'], elem['High'], elem['Low'], elem['Close'], elem['Volume']];
			writer.writerow(row)

			
if __name__ == '__main__':
	main()