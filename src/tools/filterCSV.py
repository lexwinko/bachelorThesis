import sys
import pandas as pd
import csv
import os
import re
from nltk.util import ngrams
from collections import OrderedDict

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


	grouped = txt_Import.groupby('lang',as_index=False)
	split_l = []
	split_r = []
	for x in grouped.groups:
		current = grouped.get_group(x)
		current.reset_index(drop=True,inplace=True)
		for entry in range(0,min(len(current),int(ratio))):
			split_l.append(current.iloc[entry].values)
		for entry in range(int(ratio),len(current)):
			split_r.append(current.iloc[entry].values)
	

	split_l = pd.DataFrame(split_l,columns=[ 'correctedSentence','originalSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charNGrams','wordNGrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user'])
	split_r = pd.DataFrame(split_r,columns=[ 'correctedSentence','originalSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charNGrams','wordNGrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user'])
	split_l.to_csv("output/split_"+file.split('/')[1].split('.')[0].split('_')[1]+"_lower.csv", index=False)
	split_r.to_csv("output/split_"+file.split('/')[1].split('.')[0].split('_')[1]+"_upper.csv", index=False)


def ngramModel(file, limit):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = ['correctedSentence', 'originalSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charNGrams','wordNGrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user']
	txt_Import = txt_Import[txt_Import.correctedSentence.str.contains('correctedSentence') == False]
	used_features =['charNGrams', 'wordNGrams', 'lang']
	txt_Import = txt_Import[used_features]

	grouped = txt_Import.groupby('lang',as_index=False)
	ngrams = {'German':{'charNGrams':{},'wordNGrams':{}}, 'Greek':{'charNGrams':{},'wordNGrams':{}}, 'French':{'charNGrams':{},'wordNGrams':{}}, 'Indian':{'charNGrams':{},'wordNGrams':{}}, 'Japanese':{'charNGrams':{},'wordNGrams':{}}, 'Russian':{'charNGrams':{},'wordNGrams':{}}, 'Turkish':{'charNGrams':{},'wordNGrams':{}}, 'English':{'charNGrams':{},'wordNGrams':{}}}
	for x in grouped.groups:
		current = grouped.get_group(x)
		current.reset_index(drop=True,inplace=True)
		
		for charList in current['charNGrams']:
			charList = charList.split(',')
			for entry in range(0, len(charList)):
				charList[entry] = re.sub(r" '", r'', charList[entry])
				charList[entry] = re.sub(r"\[", r'', charList[entry])
				charList[entry] = re.sub(r"\]", r'', charList[entry])
				charList[entry] = re.sub(r"'", r'', charList[entry])
				charList[entry] = charList[entry].lower()
			for chars in range(0,len(charList)):
				if not charList[chars] in ngrams[current['lang'].values[0]]['charNGrams']:
					ngrams[current['lang'].values[0]]['charNGrams'][charList[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['charNGrams'][charList[chars]] += 1
		for wordList in current['wordNGrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['lang'].values[0]]['wordNGrams']:
					ngrams[current['lang'].values[0]]['wordNGrams'][wordList[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['wordNGrams'][wordList[chars]] += 1

	ngrams_descending = OrderedDict([('German',{}),('Greek',{}),('French',{}),('Indian',{}),('Japanese',{}),('Russian',{}),('Turkish',{}),('English',{})])
	for lang in ngrams:
		ngrams_descending[lang]['charNGrams'] = sorted(ngrams[lang]['charNGrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordNGrams'] = sorted(ngrams[lang]['wordNGrams'].items(), key=lambda t: t[1], reverse=True)


	ngrams_limited = OrderedDict([('German',{}),('Greek',{}),('French',{}),('Indian',{}),('Japanese',{}),('Russian',{}),('Turkish',{}),('English',{})])
	for lang in ngrams:
		ngrams_limited[lang]['charNGrams'] = list(ngrams_descending[lang]['charNGrams'])[:max(1,int(limit))]
		ngrams_limited[lang]['wordNGrams'] = list(ngrams_descending[lang]['wordNGrams'])[:max(1,int(limit))]
	
	print(ngrams_limited)


if __name__ == "__main__":
	file = sys.argv[1]
	func = sys.argv[2]
	filters = sys.argv[3]
	if(func == 'filter'):
		print('Filtering '+ file + ' with filters: '+filters)
		filterCSV(file, filters)
	elif(func == 'split'):
		print('Splitting '+ file + ' with size: ' +filters)
		splitFile(file,filters)
	elif(func == 'ngram'):
		print('Creating ngram model '+ file + ' with limit: ' +filters)
		ngramModel(file,filters)
	print('Done')
	#print(result)