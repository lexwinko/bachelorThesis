import sys
import pandas as pd
import csv

file = ""
filters = ""

def filterCSV(file, filters):
	txt_Import = pd.read_csv(file, header=0, sep=';', usecols=['username','text'])

	usernames = []
	with open(filters, "r", encoding='utf-8') as f:
		for x in f:
			print(x)
			usernames.append(x.split()[0])
	print(usernames)

	tweets = txt_Import.loc[txt_Import['username'].isin(usernames)]

	filteredText = []
	for tweet in tweets['text']:
		print(tweet)
		tweet = tweet.replace('\xa0', ' ')
		tweet = tweet.replace('\n', ' ')
		filteredText.append(tweet)

	print(filteredText)
	print(len(filteredText))

	with open('filtered'+file, "w") as f:
		w = csv.writer(f)
		w.writerow(['text'])
		for x in filteredText:
			print(x + "\n")
			w.writerow([x])





if __name__ == "__main__":
	file = sys.argv[1]
	filters = sys.argv[2]
	print('Filtering '+ file)
	result = filterCSV(file, filters)
	print('Done')
	#print(result)