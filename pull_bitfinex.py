import sys
import csv
import time
import requests
import datetime

from datetime import timedelta

def main():

	if len(sys.argv) < 2:
		print('Usage: python pull_gdax.py <symbol>')
		return

	symbol = sys.argv[1]
	url = 'https://api.bitfinex.com/v2/candles/trade:1m:t' + symbol + '/hist'
		
	delta = datetime.timedelta(seconds = 60 * 300)

	time_data = []

	time0 = datetime.datetime.now()
	for i in range(1):
		time1 = time0
		time0 = time1 - delta

		t0 = time0.timestamp()
		t1 = time1.timestamp()

		t0 = int(t0) * 1000
		t1 = int(t1) * 1000

		time_data = [[t0, t1]] + time_data

	csv_data = []

	i = 0
	while i < len(time_data):
		timedata = time_data[i]

		params = {'start' : t0, 'end' : t1, 'limit' : 300}
		r = requests.get(url, params = params)
		
		historic_rates = r.json()

		#historic_rates = public_client.get_product_historic_rates(symbol, start = timedata[0], end = timedata[1], granularity = 60)
		#if type(historic_rates) is dict:
		#	time.sleep(1)
		#	continue

		historic_rates.sort(key = lambda r: r[0])

		for row in historic_rates:

			s = datetime.datetime.fromtimestamp(int(row[0]/1000))
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

	filename = 'bitfinex_' + symbol + '1.csv'

	#write csv
	with open(filename, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		#writer.writerow([
		#	bytes('Date',   'utf-8'),
		#	bytes('Time',   'utf-8'),
		#	bytes('Open',   'utf-8'), 
		#	bytes('High',   'utf-8'), 
		#	bytes('Low',    'utf-8'), 
		#	bytes('Close',  'utf-8'), 
		#	bytes('Volume', 'utf-8')
		#])

		for elem in csv_data:

			row = [elem['Date'],elem['Time'],elem['Open'], elem['High'], elem['Low'], elem['Close'], elem['Volume']];
			writer.writerow(row)

			
if __name__ == '__main__':
	main()