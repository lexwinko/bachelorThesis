import sys
import pandas as pd
import csv
import os

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


def splitFile(file, ratio):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = ['correctedSentence', 'originalSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charNGrams','wordNGrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user']
	txt_Import = txt_Import[txt_Import.correctedSentence.str.contains('correctedSentence') == False]

	print(txt_Import)

	grouped = txt_Import.groupby('lang',as_index=False)
	split_l = []
	split_r = []
	for x in grouped.groups:
		current = grouped.get_group(x)
		current.reset_index(drop=True,inplace=True)
		for entry in range(0,int(ratio)):
			split_l.append(current.iloc[entry].values)
		for entry in range(int(ratio),len(current)):
			split_r.append(current.iloc[entry].values)
	

	split_l = pd.DataFrame(split_l,columns=[ 'correctedSentence','originalSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charNGrams','wordNGrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user'])
	split_r = pd.DataFrame(split_r,columns=[ 'correctedSentence','originalSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charNGrams','wordNGrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user'])
	split_l.to_csv("output/split_l.csv", index=False)
	split_r.to_csv("output/split_r.csv", index=False)



if __name__ == "__main__":
	file = sys.argv[1]
	func = sys.argv[2]
	filters = sys.argv[3]
	if(func == 'filter'):
		print('Filtering '+ file + ' with filters: '+filters)
		filterCSV(file, filters)
	elif(func == 'split'):
		print('Splitting '+ file + ' with ratio: ' +filters)
		splitFile(file,filters)
	print('Done')
	#print(result)