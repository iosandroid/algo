import sys
import csv
import datetime
from dateutil import parser

arr = []

def PrepareCSV(inputFile):

	#read csv
	with open(inputFile) as csvfile:
		reader = csv.DictReader(csvfile)

		for row in reader:
			arr.append(row)


	delta = datetime.timedelta(minutes = 1)

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
	with open(outputFile, 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter = ',')
		writer.writerow(['Date','Time','Open', 'High', 'Low', 'Close', 'Volume'])

		for elem in arr:

			row = [elem['Date'],elem['Time'],elem['Open'], elem['High'], elem['Low'], elem['Close'], elem['Volume']];
			writer.writerow(row)



def main():
	for file in sys.argv[1:]:
		print 'Preparing: ' + file
		PrepareCSV(file)


if __name__ == '__main__':
	main()