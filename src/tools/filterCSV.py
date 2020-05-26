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

def tagCSV(file, family, lang):
	txt_Import = pd.read_csv(file, header=0, sep=',')

	txt_Import['family'] = family
	txt_Import['lang'] = lang

	txt_Import.to_csv('tagged_'+file.split('_')[0]+'.csv',index=False)


def splitFile(file, ratio):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = ['correctedSentence', 'originalSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user']
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
	

	split_l = pd.DataFrame(split_l,columns=[ 'correctedSentence','originalSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user'])
	split_r = pd.DataFrame(split_r,columns=[ 'correctedSentence','originalSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user'])
	split_l.to_csv("output/split_"+file.split('/')[1].split('.')[0].split('_')[1]+"_lower.csv", index=False)
	split_r.to_csv("output/split_"+file.split('/')[1].split('.')[0].split('_')[1]+"_upper.csv", index=False)


def ngramModel(file, limit, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = ['correctedSentence', 'originalSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user']
	txt_Import = txt_Import[txt_Import.correctedSentence.str.contains('correctedSentence') == False]
	used_features =['charTrigrams', 'wordBigrams', 'wordUnigrams', 'lang']
	txt_Import = txt_Import[used_features]

	grouped = txt_Import.groupby('lang',as_index=False)
	ngrams = {'German':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'French':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'Indian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'Japanese':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'Russian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'Turkish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 'English':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}}
	for x in grouped.groups:
		current = grouped.get_group(x)
		current.reset_index(drop=True,inplace=True)
		
		for charList in current['charTrigrams']:
			charList = charList.split(',')
			for entry in range(0, len(charList)):
				charList[entry] = re.sub(r" '", r'', charList[entry])
				charList[entry] = re.sub(r"\[", r'', charList[entry])
				charList[entry] = re.sub(r"\]", r'', charList[entry])
				charList[entry] = re.sub(r"'", r'', charList[entry])
				charList[entry] = charList[entry].lower()
			for chars in range(0,len(charList)):
				if not charList[chars] in ngrams[current['lang'].values[0]]['charTrigrams']:
					ngrams[current['lang'].values[0]]['charTrigrams'][charList[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['charTrigrams'][charList[chars]] += 1
		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['lang'].values[0]]['wordBigrams']:
					ngrams[current['lang'].values[0]]['wordBigrams'][wordList[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['wordBigrams'][wordList[chars]] += 1
		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['lang'].values[0]]['wordUnigrams']:
					ngrams[current['lang'].values[0]]['wordUnigrams'][wordList[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['wordUnigrams'][wordList[chars]] += 1

	ngrams_descending = OrderedDict([('German',{}),('Greek',{}),('French',{}),('Indian',{}),('Japanese',{}),('Russian',{}),('Turkish',{}),('English',{})])
	for lang in ngrams:
		ngrams_descending[lang]['charTrigrams'] = sorted(ngrams[lang]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordBigrams'] = sorted(ngrams[lang]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordUnigrams'] = sorted(ngrams[lang]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)


	ngrams_limited = OrderedDict([('German',{}),('Greek',{}),('French',{}),('Indian',{}),('Japanese',{}),('Russian',{}),('Turkish',{}),('English',{})])
	for lang in ngrams:
		ngrams_limited[lang]['charTrigrams'] = list(ngrams_descending[lang]['charTrigrams'])[:max(1,int(limit))]
		ngrams_limited[lang]['wordBigrams'] = list(ngrams_descending[lang]['wordBigrams'])[:max(1,int(limit))]
		ngrams_limited[lang]['wordUnigrams'] = list(ngrams_descending[lang]['wordUnigrams'])[:max(1,int(limit))]
	
	for lang in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+lang+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for entry in range(0,min(len(ngrams_limited[lang]['charTrigrams']),len(ngrams_limited[lang]['wordBigrams']),len(ngrams_limited[lang]['wordUnigrams']),int(limit))):
				writer.writerow({'charTrigrams': ngrams_limited[lang]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[lang]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[lang]['wordUnigrams'][entry][0]})

def createClassifierFile(file,filters):
	data = pd.read_csv(file, header=0, sep=',')
	lang = ['French', 'German', 'Greek', 'English', 'Indian', 'Japanese', 'Russian', 'Turkish']
	ngrams = ['charTrigrams','wordBigrams','wordUnigrams', 'lang']
	features_similarity = [	'charTrigrams_similarity_French',
							'wordBigrams_similarity_French',
							'wordUnigrams_similarity_French',
							'charTrigrams_similarity_German',
							'wordBigrams_similarity_German',
							'wordUnigrams_similarity_German',
							'charTrigrams_similarity_Greek',
							'wordBigrams_similarity_Greek',
							'wordUnigrams_similarity_Greek',
							'charTrigrams_similarity_Indian',
							'wordBigrams_similarity_Indian',
							'wordUnigrams_similarity_Indian',
							'charTrigrams_similarity_Russian',
							'wordBigrams_similarity_Russian',
							'wordUnigrams_similarity_Russian',
							'charTrigrams_similarity_Japanese',
							'wordBigrams_similarity_Japanese',
							'wordUnigrams_similarity_Japanese',
							'charTrigrams_similarity_Turkish',
							'wordBigrams_similarity_Turkish',
							'wordUnigrams_similarity_Turkish',
							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English']

	for feature in features_similarity:
		data[feature] = 0


	ngram_data = {	'French':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'German':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Greek':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'English':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Indian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Japanese':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Russian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Turkish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}}
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
		
		for entry in raw_ngrams['charTrigrams']:
			ngram_data[language]['charTrigrams'].add(str(entry))
		for entry in raw_ngrams['wordBigrams']:
			ngram_data[language]['wordBigrams'].add(str(entry))
		for entry in raw_ngrams['wordUnigrams']:
			ngram_data[language]['wordUnigrams'].add(str(entry))

	grouped = data[ngrams].groupby('lang',as_index=False)
	for x in grouped.groups:
		current = grouped.get_group(x)
		print('Current language: '+x)
		for index, row in current.iterrows():
			print(index)
			ngram_current = {'charTrigrams':NGram(), 'wordBigrams':NGram(), 'wordUnigrams':NGram()}
			ngramlist = row['charTrigrams'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['charTrigrams'].add(str(ngramlist[entry]))
			ngramlist = row['wordBigrams'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['wordBigrams'].add(str(ngramlist[entry]))
			ngramlist = row['wordUnigrams'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['wordUnigrams'].add(str(ngramlist[entry]))

			similarity = {	'French':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'German':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Greek':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'English':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Indian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Japanese':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Russian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Turkish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}}

			for language in lang:
				similarity[language] = getNGramSimilarity(ngram_current, ngram_data[language])
				data.loc[index, 'charTrigrams_similarity_'+str(language)] = similarity[language]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(language)] = similarity[language]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(language)] = similarity[language]['wordUnigrams']

	data.to_csv("output/classification_data_"+filters+".csv", index=False)

def getNGramSimilarity(ngrams, data):
	#print("similarity")
	#print('ngrams', len(ngrams['charTrigrams']), len(ngrams['wordBigrams']))
	#print('data',len(data['charTrigrams']), len(data['wordBigrams']))
	intersection = {'charTrigrams': list(ngrams['charTrigrams'].intersection(data['charTrigrams'])), 'wordBigrams':list(ngrams['wordBigrams'].intersection(data['wordBigrams'])), 'wordUnigrams':list(ngrams['wordUnigrams'].intersection(data['wordUnigrams']))}
	union = {'charTrigrams': list(ngrams['charTrigrams'].union(data['charTrigrams'])), 'wordBigrams':list(ngrams['wordBigrams'].union(data['wordBigrams'])), 'wordUnigrams':list(ngrams['wordUnigrams'].union(data['wordUnigrams']))}
	#print(len(intersection['charTrigrams']), len(intersection['wordBigrams']), len(union['charTrigrams']), len(union['wordBigrams']))
	similariy = {'charTrigrams': NGram.ngram_similarity(len(intersection['charTrigrams']), len(union['charTrigrams'])), 'wordBigrams': NGram.ngram_similarity(len(intersection['wordBigrams']), len(union['wordBigrams'])), 'wordUnigrams': NGram.ngram_similarity(len(intersection['wordUnigrams']), len(union['wordUnigrams']))}
	return similariy


if __name__ == "__main__":
	file = sys.argv[1]
	if(len(sys.argv) > 2):
		func = sys.argv[2]
	if(len(sys.argv) > 3):
		filters = sys.argv[3]
	if(len(sys.argv) > 4):
		arg2 = sys.argv[4]
	if(func == 'tag'):
		print('Tagging '+ file + ' with filters: '+filters+', '+arg2)
		tagCSV(file, filters, arg2)
	elif(func == 'split'):
		print('Splitting '+ file + ' with size: ' +filters)
		splitFile(file,filters)
	elif(func == 'ngram'):
		print('Creating ngram model '+ file + ' with limit: ' +filters)
		ngramModel(file,filters,arg2)
	elif(func == 'classifier'):
		print('Creating classification file for ' +filters)
		createClassifierFile(file,filters)
	print('Done')
	#print(result)