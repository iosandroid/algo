import sys
import csv
import datetime
from dateutil import parser

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

def FillGapsInCSV(inputFile):

	arr = []

	lhs, rhs = inputFile.split('.')
	granularity = lhs[-3:] if lhs[-3:-1].isdigit() else lhs[-2:]

	print(granularity)

	#read csv
	with open(inputFile) as csvfile:
		reader = csv.DictReader(csvfile)

		for row in reader:
			arr.append(row)

	delta = datetime.timedelta(seconds = switch_granularity(granularity))

	for i in range(len(arr)):

		d_temp = parser.parse(arr[i]['Date']).date()
		t_temp = parser.parse(arr[i]['Time']).time()

		stamp = datetime.datetime(
			year = d_temp.year, 
			month = d_temp.month, 
			day = d_temp.day, 
			hour = t_temp.hour, 
			minute = t_temp.minute
		)

		arr[i]['stamp'] = stamp

	for i in range(len(arr)-1):

		s0 = arr[i]['stamp']
		s1 = arr[i+1]['stamp']

		s = s0 + delta
		row = arr[i]

		while s < s1:
			new_row = {
				'stamp'  : s, 
				'Date'   : s.date().strftime('%Y.%m.%d'), 
				'Time'   : s.time().strftime('%H:%M'), 
				'Open'   : row['Open'], 
				'High'   : row['High'], 
				'Low'    : row['Low'], 
				'Close'  : row['Close'], 
				'Volume' : row['Volume']
			}

			arr.append(new_row)
			s = s + delta	

	arr.sort(key = lambda r: r['stamp'])

	for i in range(len(arr)):
		arr[i].pop('stamp')

	outputFile = inputFile

	#write csv
	with open(outputFile, 'w') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(['Date','Time','Open', 'High', 'Low', 'Close', 'Volume'])

		for elem in arr:

			row = [elem['Date'],elem['Time'],elem['Open'], elem['High'], elem['Low'], elem['Close'], elem['Volume']];
			writer.writerow(row)



def main():
	for file in sys.argv[1:]:

		print('Validating: ' + file)
		FillGapsInCSV(file)

	data = {}

	first_rows = []
	last_rows = []

	for file in sys.argv[1:]:

		with open(file) as csvfile:
			reader = csv.DictReader(csvfile)

			data[file] = []

			first_row = None
			last_row = None
			
			for row in reader:

				if first_row is None:
					first_row = row

				last_row = row

				d_temp = parser.parse(row['Date']).date()
				t_temp = parser.parse(row['Time']).time()

				stamp = datetime.datetime(
					year   = d_temp.year, 
					month  = d_temp.month, 
					day    = d_temp.day, 
					hour   = t_temp.hour, 
					minute = t_temp.minute
				)

				row['stamp'] = stamp				

				data[file].append(row)

			first_rows.append(first_row)
			last_rows.append(last_row)
	
	start_stamp = max(first_rows, key = lambda r: r['stamp'])['stamp']
	end_stamp = min(last_rows, key = lambda r: r['stamp'])['stamp']

	print(start_stamp)
	print(end_stamp)

	for file in sys.argv[1:]:

		olddata = data[file]
		newdata = [row for row in olddata if (row['stamp'] >= start_stamp) and (row['stamp'] <= end_stamp)]

		data[file] = newdata

	for file in sys.argv[1:]:
		for row in data[file]:
			row.pop('stamp')

	for file in sys.argv[1:]:

		with open(file, 'w') as csvfile:

			writer = csv.writer(csvfile, delimiter = ',')
			writer.writerow(['Date','Time','Open', 'High', 'Low', 'Close', 'Volume'])

			for row in data[file]:

				new_row = [row['Date'],row['Time'],row['Open'],row['High'],row['Low'],row['Close'],row['Volume']];
				writer.writerow(new_row)
			
if __name__ == '__main__':
	main()