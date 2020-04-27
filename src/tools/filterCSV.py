import sys
import pandas as pd
import csv
import os
import re
from ngram import NGram
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
	
	for lang in ngrams_limited:
		filename = "output/ngrams_"+lang+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charNGrams', 'wordNGrams']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for entry in range(0,min(len(ngrams_limited[lang]['charNGrams']),len(ngrams_limited[lang]['wordNGrams']),int(limit))):
				writer.writerow({'charNGrams': ngrams_limited[lang]['charNGrams'][entry][0], 'wordNGrams': ngrams_limited[lang]['wordNGrams'][entry][0]})

def createClassifierFile(file):
	data = pd.read_csv(file, header=0, sep=',')
	lang = ['French', 'German', 'Greek', 'English', 'Indian', 'Japanese', 'Russian', 'Turkish']
	ngrams = ['charNGrams','wordNGrams', 'lang']
	features_similarity = [	'charNGrams_similarity_French',
							'wordNGrams_similarity_French',
							'charNGrams_similarity_German',
							'wordNGrams_similarity_German',
							'charNGrams_similarity_Greek',
							'wordNGrams_similarity_Greek',
							'charNGrams_similarity_Indian',
							'wordNGrams_similarity_Indian',
							'charNGrams_similarity_Russian',
							'wordNGrams_similarity_Russian',
							'charNGrams_similarity_Japanese',
							'wordNGrams_similarity_Japanese',
							'charNGrams_similarity_Turkish',
							'wordNGrams_similarity_Turkish',
							'charNGrams_similarity_English',
							'wordNGrams_similarity_English']

	for feature in features_similarity:
		data[feature] = 0


	ngram_data = {	'French':{'charNGrams':NGram(),'wordNGrams':NGram()}, 
					'German':{'charNGrams':NGram(),'wordNGrams':NGram()}, 
					'Greek':{'charNGrams':NGram(),'wordNGrams':NGram()}, 
					'English':{'charNGrams':NGram(),'wordNGrams':NGram()},
					'Indian':{'charNGrams':NGram(),'wordNGrams':NGram()}, 
					'Japanese':{'charNGrams':NGram(),'wordNGrams':NGram()}, 
					'Russian':{'charNGrams':NGram(),'wordNGrams':NGram()}, 
					'Turkish':{'charNGrams':NGram(),'wordNGrams':NGram()}}
	for language in lang:
		raw_ngrams = []
		if(language == 'French'):
			raw_ngrams = pd.read_csv('../../data/processed/France/ngrams_French.csv', header=0)
		elif(language == 'German'):
			raw_ngrams = pd.read_csv('../../data/processed/Germany/ngrams_German.csv', header=0)
		elif(language == 'Greek'):
			raw_ngrams = pd.read_csv('../../data/processed/Greece/ngrams_Greek.csv', header=0)
		elif(language == 'Indian'):
			raw_ngrams = pd.read_csv('../../data/processed/India/ngrams_Indian.csv', header=0)
		elif(language == 'Japanese'):
			raw_ngrams = pd.read_csv('../../data/processed/Japan/ngrams_Japanese.csv', header=0)
		elif(language == 'Russian'):
			raw_ngrams = pd.read_csv('../../data/processed/Russia/ngrams_Russian.csv', header=0)
		elif(language == 'Turkish'):
			raw_ngrams = pd.read_csv('../../data/processed/Turkey/ngrams_Turkish.csv', header=0)
		else:
			raw_ngrams = pd.read_csv('../../data/processed/Native/ngrams_English.csv', header=0)
		
		for entry in raw_ngrams['charNGrams']:
			ngram_data[language]['charNGrams'].add(str(entry))
		for entry in raw_ngrams['wordNGrams']:
			ngram_data[language]['wordNGrams'].add(str(entry))

	grouped = data[ngrams].groupby('lang',as_index=False)
	for x in grouped.groups:
		current = grouped.get_group(x)
		print('Current language: '+x)
		for index, row in current.iterrows():
			print(index)
			ngram_current = {'charNGrams':NGram(), 'wordNGrams':NGram()}
			ngramlist = row['charNGrams'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['charNGrams'].add(str(ngramlist[entry]))
			ngramlist = row['wordNGrams'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['wordNGrams'].add(str(ngramlist[entry]))

			similarity = {	'French':{'charNGrams':0,'wordNGrams':0}, 
						'German':{'charNGrams':0,'wordNGrams':0}, 
						'Greek':{'charNGrams':0,'wordNGrams':0}, 
						'English':{'charNGrams':0,'wordNGrams':0},
						'Indian':{'charNGrams':0,'wordNGrams':0}, 
						'Japanese':{'charNGrams':0,'wordNGrams':0}, 
						'Russian':{'charNGrams':0,'wordNGrams':0}, 
						'Turkish':{'charNGrams':0,'wordNGrams':0}}

			for language in lang:
				similarity[language] = getNGramSimilarity(ngram_current, ngram_data[language])
				data.loc[index, 'charNGrams_similarity_'+str(language)] = similarity[language]['charNGrams']
				data.loc[index, 'wordNGrams_similarity_'+str(language)] = similarity[language]['wordNGrams']

	data.to_csv("output/classification_data.csv", index=False)

def getNGramSimilarity(ngrams, data):
	#print("similarity")
	#print('ngrams', len(ngrams['charNGrams']), len(ngrams['wordNGrams']))
	#print('data',len(data['charNGrams']), len(data['wordNGrams']))
	intersection = {'charNGrams': list(ngrams['charNGrams'].intersection(data['charNGrams'])), 'wordNGrams':list(ngrams['wordNGrams'].intersection(data['wordNGrams']))}
	union = {'charNGrams': list(ngrams['charNGrams'].union(data['charNGrams'])), 'wordNGrams':list(ngrams['wordNGrams'].union(data['wordNGrams']))}
	#print(len(intersection['charNGrams']), len(intersection['wordNGrams']), len(union['charNGrams']), len(union['wordNGrams']))
	similariy = {'charNGrams': NGram.ngram_similarity(len(intersection['charNGrams']), len(union['charNGrams'])), 'wordNGrams': NGram.ngram_similarity(len(intersection['wordNGrams']), len(union['wordNGrams']))}
	return similariy


if __name__ == "__main__":
	file = sys.argv[1]
	if(len(sys.argv) > 2):
		func = sys.argv[2]
	if(len(sys.argv) > 3):
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
	elif(func == 'classifier'):
		print('Creating classification file')
		createClassifierFile(file)
	print('Done')
	#print(result)