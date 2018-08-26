import sys
import csv
import time
import requests
import datetime

from datetime import timedelta

def main():

	symbol = 'BTCUSD'
	delta = datetime.timedelta(seconds = 60)

	url = 'https://api.bitfinex.com/v2/trades/t' + symbol + '/hist'

	count = 60	
	limit = 400

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

		historic_rates.sort(key = lambda r: r[1])
			
		for row in historic_rates:

			s = datetime.datetime.fromtimestamp(int(row[1]/1000))
			new_row = {
				'<Unknown>' : row[0],
				'Date'      : s.date().strftime('%Y.%m.%d'),
				'Time'      : s.time().strftime('%H:%M:%S'),
				'Amount'    : row[2],
				'Price'     : row[3]
			}

			csv_data.append(new_row)

		i = i + 1

	filename = 'bitfinex_' + symbol + '_trades.csv'

	#write csv
	with open(filename, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(['Date', 'Time', 'Amount', 'Price', '<Unknown>'])

		for elem in csv_data:

			row = [elem['Date'], elem['Time'], elem['Amount'],elem['Price'], elem['<Unknown>']];
			writer.writerow(row)


if __name__ == '__main__':
	main()