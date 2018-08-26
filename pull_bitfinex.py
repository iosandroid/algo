import sys
import csv
import time
import requests
import datetime

from datetime import timedelta

def switch_granularity(x):
	return {
		'1m' : 60,
		'5m' : 60*5,
		'15m': 60*15,
		'30m': 60*30,
		'1h' : 60*60,
		'3h' : 60*60*3,
		'6h' : 60*60*6,
		'12h': 60*60*12,
		'1D' : 60*60*24,
		'7D' : 60*60*24*7,
		'14D': 60*60*24*14
	}[x]

def main():

	if len(sys.argv) < 2:
		print('Usage: python pull_gdax.py <symbol>')
		return

	symbol = sys.argv[1]
	granularity = sys.argv[2] if len(sys.argv) > 2 else '1m'
	size = sys.argv[3] if len(sys.argv) > 3 else 10000

	url = 'https://api.bitfinex.com/v2/candles/trade:' + granularity + ':t' + symbol + '/hist'

	limit = 400
	count = int(size / limit)
	delta = datetime.timedelta(seconds = switch_granularity(granularity) * limit)

	time_data = []

	time0 = datetime.datetime.now()
	for i in range(count):
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

		params = {'start' : timedata[0], 'end' : timedata[1], 'limit' : limit}
		r = requests.get(url, params = params)
		
		historic_rates = r.json()

		time.sleep(4)

		if 'error' in historic_rates:

			print(historic_rates)
			time.sleep(60)

			continue

		print(i)


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

	filename = 'bitfinex_' + symbol + granularity + '.csv'

	#write csv
	with open(filename, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

		for elem in csv_data:

			row = [elem['Date'],elem['Time'],elem['Open'], elem['High'], elem['Low'], elem['Close'], elem['Volume']];
			writer.writerow(row)

			
if __name__ == '__main__':
	main()