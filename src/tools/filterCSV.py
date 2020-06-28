import sys
import pandas as pd
import csv
import os
import re
from ngram import NGram
from nltk.util import ngrams
from collections import OrderedDict

from sklearn.feature_extraction.text import TfidfVectorizer 

file = ""
filters = ""

def tagCSV(file, family, lang):
	txt_Import = pd.read_csv(file, header=0, sep=',')

	txt_Import['family'] = family
	txt_Import['lang'] = lang

	txt_Import.to_csv('tagged_'+file.split('_')[0]+'.csv',index=False)


def splitFile(file, ratio, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	if(origin == 'reddit'):
		txt_Import.columns = ['user', 'subreddit', 'post', 'langFam', 'lang']
		txt_Import = txt_Import[txt_Import.subreddit.str.contains('subreddit') == False]
	else:
		txt_Import.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user', 'category']
		txt_Import = txt_Import[txt_Import.filteredSentence.str.contains('filteredSentence') == False]
	txt_Import = txt_Import.sample(frac=1,random_state=42).reset_index(drop=True)

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
	
	if(origin == 'reddit'):
		fileCol = ['user', 'subreddit', 'post', 'langFam', 'lang']
	else:
		fileCol = [ 'correctedSentence','originalSentence','filteredSentence','stemmedSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user', 'category']

	split_l = pd.DataFrame(split_l,columns=fileCol)
	split_r = pd.DataFrame(split_r,columns=fileCol)
	split_l.to_csv("output/split_"+origin+"_lower_lang.csv", index=False)
	split_r.to_csv("output/split_"+origin+"_upper_lang.csv", index=False)

	grouped = txt_Import.groupby('langFam',as_index=False)
	split_l = []
	split_r = []
	for x in grouped.groups:
		current = grouped.get_group(x)
		current.reset_index(drop=True,inplace=True)
		for entry in range(0,min(len(current),int(ratio)*2)):
			split_l.append(current.iloc[entry].values)
		for entry in range(int(ratio)*2,len(current)):
			split_r.append(current.iloc[entry].values)
	

	split_l = pd.DataFrame(split_l,columns=fileCol)
	split_r = pd.DataFrame(split_r,columns=fileCol)
	split_l.to_csv("output/split_"+origin+"_lower_family.csv", index=False)
	split_r.to_csv("output/split_"+origin+"_upper_family.csv", index=False)

	if(origin == 'twitter'):
		grouped = txt_Import.groupby('category',as_index=False)
		split_l = []
		split_r = []
		for x in grouped.groups:
			current = grouped.get_group(x)
			current.reset_index(drop=True,inplace=True)
			for entry in range(0,min(len(current),int(ratio)*4)):
				split_l.append(current.iloc[entry].values)
			for entry in range(int(ratio)*4,len(current)):
				split_r.append(current.iloc[entry].values)
		

		split_l = pd.DataFrame(split_l,columns=fileCol)
		split_r = pd.DataFrame(split_r,columns=fileCol)
		split_l.to_csv("output/split_"+origin+"_lower_category.csv", index=False)
		split_r.to_csv("output/split_"+origin+"_upper_category.csv", index=False)


def ngramModel(file, limit, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = ['correctedSentence', 'originalSentence', 'filteredSentence', 'stemmedSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user', 'category']
	txt_Import = txt_Import[txt_Import.filteredSentence.str.contains('filteredSentence') == False]
	used_features =['charTrigrams', 'wordBigrams', 'wordUnigrams', 'lang', 'langFam', 'category']
	txt_Import = txt_Import[used_features]

	grouped = txt_Import.groupby('lang',as_index=False)

	# if(origin == 'twitter'):
	# 	ngrams = {
	# 			'German':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'French':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Indian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Japanese':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Russian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Turkish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'English':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}
	# 			}
	# else:
	ngrams = {
			'German':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'French':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Indian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Japanese':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Russian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Turkish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Bulgarian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Croatian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Czech':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Lithuanian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Polish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Serbian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Slovene':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Finnish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Dutch':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Norwegian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Swedish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Italian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Spanish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Portugese':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Romanian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Estonian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Hungarian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'English':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}
			}
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

	# if(origin == 'twitter'):
	# 	ngrams_descending = OrderedDict([
	# 					('German',{}),
	# 					('Greek',{}),
	# 					('French',{}),
	# 					('Indian',{}),
	# 					('Japanese',{}),
	# 					('Russian',{}),
	# 					('Turkish',{}),
	# 					('English',{})
	# 					])
	# else:
	ngrams_descending = OrderedDict([
					('German',{}),
					('Greek',{}),
					('French',{}),
					('Indian',{}),
					('Japanese',{}),
					('Russian',{}),
					('Turkish',{}),
					('Bulgarian',{}),
					('Croatian',{}),
					('Czech',{}),
					('Lithuanian',{}),
					('Polish',{}),
					('Serbian',{}),
					('Slovene',{}),
					('Finnish',{}),
					('Dutch',{}),
					('Norwegian',{}),
					('Swedish',{}),
					('Italian',{}),
					('Spanish',{}),
					('Portugese',{}),
					('Romanian',{}),
					('Estonian',{}),
					('Hungarian',{}),
					('English',{})
					])
	for lang in ngrams:
		ngrams_descending[lang]['charTrigrams'] = sorted(ngrams[lang]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordBigrams'] = sorted(ngrams[lang]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordUnigrams'] = sorted(ngrams[lang]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)

	# if(origin == 'twitter'):
	# 	ngrams_limited = OrderedDict([
	# 					('German',{}),
	# 					('Greek',{}),
	# 					('French',{}),
	# 					('Indian',{}),
	# 					('Japanese',{}),
	# 					('Russian',{}),
	# 					('Turkish',{}),
	# 					('English',{})
	# 					])
	# else:
	ngrams_limited = OrderedDict([
					('German',{}),
					('Greek',{}),
					('French',{}),
					('Indian',{}),
					('Japanese',{}),
					('Russian',{}),
					('Turkish',{}),
					('Bulgarian',{}),
					('Croatian',{}),
					('Czech',{}),
					('Lithuanian',{}),
					('Polish',{}),
					('Serbian',{}),
					('Slovene',{}),
					('Finnish',{}),
					('Dutch',{}),
					('Norwegian',{}),
					('Swedish',{}),
					('Italian',{}),
					('Spanish',{}),
					('Portugese',{}),
					('Romanian',{}),
					('Estonian',{}),
					('Hungarian',{}),
					('English',{})
					])
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


	grouped_family = txt_Import.groupby('langFam',as_index=False)
	# if(origin == 'twitter'):
	# 	ngrams = {
	# 			'Turkic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Romance':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Germanic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
	# 			'Balto-Slavic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
	# 			'Japonic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
	# 			'Indo-Aryan':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}
	# 			}
	# else:
	ngrams = {
			'Uralic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Turkic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Romance':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Germanic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
			'Balto-Slavic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Japonic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
			'Indo-Aryan':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}
			}
	for x in grouped_family.groups:
		current = grouped_family.get_group(x)
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
				if not charList[chars] in ngrams[current['langFam'].values[0]]['charTrigrams']:
					ngrams[current['langFam'].values[0]]['charTrigrams'][charList[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['charTrigrams'][charList[chars]] += 1
		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['langFam'].values[0]]['wordBigrams']:
					ngrams[current['langFam'].values[0]]['wordBigrams'][wordList[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['wordBigrams'][wordList[chars]] += 1
		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['langFam'].values[0]]['wordUnigrams']:
					ngrams[current['langFam'].values[0]]['wordUnigrams'][wordList[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['wordUnigrams'][wordList[chars]] += 1
	# if(origin == 'twitter'):		
	# 	ngrams_descending = OrderedDict([
	# 					('Turkic',{}),
	# 					('Romance',{}),
	# 					('Greek',{}),
	# 					('Germanic',{}),
	# 					('Balto-Slavic',{}),
	# 					('Japonic',{}),
	# 					('Indo-Aryan',{})
	# 					])
	# else:
	ngrams_descending = OrderedDict([
					('Uralic',{}),
					('Turkic',{}),
					('Romance',{}),
					('Greek',{}),
					('Germanic',{}),
					('Balto-Slavic',{}),
					('Japonic',{}),
					('Indo-Aryan',{})
					])
	for family in ngrams:
		ngrams_descending[family]['charTrigrams'] = sorted(ngrams[family]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[family]['wordBigrams'] = sorted(ngrams[family]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[family]['wordUnigrams'] = sorted(ngrams[family]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)

	# if(origin == 'twitter'):
	# 	ngrams_limited = OrderedDict([
	# 					('Turkic',{}),
	# 					('Romance',{}),
	# 					('Greek',{}),
	# 					('Germanic',{}),
	# 					('Balto-Slavic',{}),
	# 					('Japonic',{}),
	# 					('Indo-Aryan',{})
	# 					])
	# else:
	ngrams_limited = OrderedDict([
					('Uralic',{}),
					('Turkic',{}),
					('Romance',{}),
					('Greek',{}),
					('Germanic',{}),
					('Balto-Slavic',{}),
					('Japonic',{}),
					('Indo-Aryan',{})
					])
	for family in ngrams:
		ngrams_limited[family]['charTrigrams'] = list(ngrams_descending[family]['charTrigrams'])[:max(1,int(limit))]
		ngrams_limited[family]['wordBigrams'] = list(ngrams_descending[family]['wordBigrams'])[:max(1,int(limit))]
		ngrams_limited[family]['wordUnigrams'] = list(ngrams_descending[family]['wordUnigrams'])[:max(1,int(limit))]
	
	for family in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+family+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for entry in range(0,min(len(ngrams_limited[family]['charTrigrams']),len(ngrams_limited[family]['wordBigrams']),len(ngrams_limited[family]['wordUnigrams']),int(limit))):
				writer.writerow({'charTrigrams': ngrams_limited[family]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[family]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[family]['wordUnigrams'][entry][0]})

	grouped_category = txt_Import.groupby('category',as_index=False)
	ngrams = {
		'ArtCul':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
		'BuiTecSci':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
		'Pol':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
		'SocSoc':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}},
		'European':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}, 
		'NonEuropean':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{}}
		}
	for x in grouped_category.groups:
		current = grouped_category.get_group(x)
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
				if not charList[chars] in ngrams[current['category'].values[0]]['charTrigrams']:
					ngrams[current['category'].values[0]]['charTrigrams'][charList[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['charTrigrams'][charList[chars]] += 1
		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['category'].values[0]]['wordBigrams']:
					ngrams[current['category'].values[0]]['wordBigrams'][wordList[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['wordBigrams'][wordList[chars]] += 1
		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
			for chars in range(0,len(wordList)):
				if not wordList[chars] in ngrams[current['category'].values[0]]['wordUnigrams']:
					ngrams[current['category'].values[0]]['wordUnigrams'][wordList[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['wordUnigrams'][wordList[chars]] += 1

	ngrams_descending = OrderedDict([
					('ArtCul',{}),
					('BuiTecSci',{}),
					('Pol',{}),
					('SocSoc',{}),
					('European',{}),
					('NonEuropean',{})
					])
	for category in ngrams:
		ngrams_descending[category]['charTrigrams'] = sorted(ngrams[category]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[category]['wordBigrams'] = sorted(ngrams[category]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[category]['wordUnigrams'] = sorted(ngrams[category]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)


	ngrams_limited = OrderedDict([
					('ArtCul',{}),
					('BuiTecSci',{}),
					('Pol',{}),
					('SocSoc',{}),
					('European',{}),
					('NonEuropean',{})
					])
	for category in ngrams:
		ngrams_limited[category]['charTrigrams'] = list(ngrams_descending[category]['charTrigrams'])[:max(1,int(limit))]
		ngrams_limited[category]['wordBigrams'] = list(ngrams_descending[category]['wordBigrams'])[:max(1,int(limit))]
		ngrams_limited[category]['wordUnigrams'] = list(ngrams_descending[category]['wordUnigrams'])[:max(1,int(limit))]
	
	for category in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+category+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams']
			writer = csv.DictWriter(f, fieldnames=fieldnames)
			writer.writeheader()
			for entry in range(0,min(len(ngrams_limited[category]['charTrigrams']),len(ngrams_limited[category]['wordBigrams']),len(ngrams_limited[category]['wordUnigrams']),int(limit))):
				writer.writerow({'charTrigrams': ngrams_limited[category]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[category]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[category]['wordUnigrams'][entry][0]})


def createClassifierFile(file,filters):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user', 'category']
	data = data[data.filteredSentence.str.contains('filteredSentence') == False]

	if(filters == 'reddit'):
		extension = 'reddit'
		category= [
				'European',
				'NonEuropean'
				]
	elif(filters == 'twitter'):
		extension = 'twitter'
		category= [
				'ArtCul',
				'BuiTecSci',
				'Pol',
				'SocSoc'
				]
	else:
		extension = 'combined'
		category = [
				'ArtCul',
				'BuiTecSci',
				'Pol',
				'SocSoc',
				'European',
				'NonEuropean'
		]

	lang = [
			'French', 
			'German', 
			'Greek', 
			'English', 
			'Indian', 
			'Japanese', 
			'Russian', 
			'Turkish', 
			'Bulgarian',
			'Croatian',
			'Czech',
			'Lithuanian',
			'Polish',
			'Serbian',
			'Slovene',
			'Finnish',
			'Dutch',
			'Norwegian',
			'Swedish',
			'Italian',
			'Spanish',
			'Portugese',
			'Romanian',
			'Estonian',
			'Hungarian'
			]
	family= [
			'Balto-Slavic',
			'Germanic',
			'Indo-Aryan',
			'Japonic',
			'Romance',
			'Turkic',
			'Uralic',
			'Greek'
			]

	ngrams = ['charTrigrams','wordBigrams','wordUnigrams', 'lang', 'langFam', 'category']
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
							'charTrigrams_similarity_Bulgarian',
							'wordBigrams_similarity_Bulgarian',
							'wordUnigrams_similarity_Bulgarian',
							'charTrigrams_similarity_Croatian',
							'wordBigrams_similarity_Croatian',
							'wordUnigrams_similarity_Croatian',
							'charTrigrams_similarity_Czech',
							'wordBigrams_similarity_Czech',
							'wordUnigrams_similarity_Czech',
							'charTrigrams_similarity_Lithuanian',
							'wordBigrams_similarity_Lithuanian',
							'wordUnigrams_similarity_Lithuanian',
							'charTrigrams_similarity_Polish',
							'wordBigrams_similarity_Polish',
							'wordUnigrams_similarity_Polish',
							'charTrigrams_similarity_Serbian',
							'wordBigrams_similarity_Serbian',
							'wordUnigrams_similarity_Serbian',
							'charTrigrams_similarity_Slovene',
							'wordBigrams_similarity_Slovene',
							'wordUnigrams_similarity_Slovene',
							'charTrigrams_similarity_Finnish',
							'wordBigrams_similarity_Finnish',
							'wordUnigrams_similarity_Finnish',
							'charTrigrams_similarity_Dutch',
							'wordBigrams_similarity_Dutch',
							'wordUnigrams_similarity_Dutch',
							'charTrigrams_similarity_Norwegian',
							'wordBigrams_similarity_Norwegian',
							'wordUnigrams_similarity_Norwegian',
							'charTrigrams_similarity_Swedish',
							'wordBigrams_similarity_Swedish',
							'wordUnigrams_similarity_Swedish',
							'charTrigrams_similarity_Italian',
							'wordBigrams_similarity_Italian',
							'wordUnigrams_similarity_Italian',
							'charTrigrams_similarity_Spanish',
							'wordBigrams_similarity_Spanish',
							'wordUnigrams_similarity_Spanish',
							'charTrigrams_similarity_Portugese',
							'wordBigrams_similarity_Portugese',
							'wordUnigrams_similarity_Portugese',
							'charTrigrams_similarity_Romanian',
							'wordBigrams_similarity_Romanian',
							'wordUnigrams_similarity_Romanian',
							'charTrigrams_similarity_Estonian',
							'wordBigrams_similarity_Estonian',
							'wordUnigrams_similarity_Estonian',
							'charTrigrams_similarity_Hungarian',
							'wordBigrams_similarity_Hungarian',
							'wordUnigrams_similarity_Hungarian',
							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English',
							'charTrigrams_similarity_Balto-Slavic',
							'wordBigrams_similarity_Balto-Slavic',
							'wordUnigrams_similarity_Balto-Slavic',
							'charTrigrams_similarity_Germanic',
							'wordBigrams_similarity_Germanic',
							'wordUnigrams_similarity_Germanic',
							'charTrigrams_similarity_Romance',
							'wordBigrams_similarity_Romance',
							'wordUnigrams_similarity_Romance',
							'charTrigrams_similarity_Japonic',
							'wordBigrams_similarity_Japonic',
							'wordUnigrams_similarity_Japonic',
							'charTrigrams_similarity_Turkic',
							'wordBigrams_similarity_Turkic',
							'wordUnigrams_similarity_Turkic',
							'charTrigrams_similarity_Uralic',
							'wordBigrams_similarity_Uralic',
							'wordUnigrams_similarity_Uralic',
							'charTrigrams_similarity_Indo-Aryan',
							'wordBigrams_similarity_Indo-Aryan',
							'wordUnigrams_similarity_Indo-Aryan',
							'charTrigrams_similarity_European',
							'wordBigrams_similarity_European',
							'wordUnigrams_similarity_European',
							'charTrigrams_similarity_NonEuropean',
							'wordBigrams_similarity_NonEuropean',
							'wordUnigrams_similarity_NonEuropean',
							'charTrigrams_similarity_ArtCul',
							'wordBigrams_similarity_ArtCul',
							'wordUnigrams_similarity_ArtCul',
							'charTrigrams_similarity_BuiTecSci',
							'wordBigrams_similarity_BuiTecSci',
							'wordUnigrams_similarity_BuiTecSci',
							'charTrigrams_similarity_Pol',
							'wordBigrams_similarity_Pol',
							'wordUnigrams_similarity_Pol',
							'charTrigrams_similarity_SocSoc',
							'wordBigrams_similarity_SocSoc',
							'wordUnigrams_similarity_SocSoc'
							]

	for feature in features_similarity:
		data[feature] = 0


	ngram_data = {	
					'French':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'German':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Greek':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'English':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Indian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Japanese':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Russian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}, 
					'Turkish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Bulgarian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Croatian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Czech':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Lithuanian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Polish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Serbian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Slovene':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Finnish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Dutch':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Norwegian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Swedish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Italian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Spanish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Portugese':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Romanian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Estonian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Hungarian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Balto-Slavic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Germanic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Romance':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Japonic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Turkic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Uralic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Indo-Aryan':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'European':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'NonEuropean':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'ArtCul':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'BuiTecSci':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'Pol':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()},
					'SocSoc':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram()}
					}
	for language in lang:
		raw_ngrams = []
		raw_ngrams = pd.read_csv('../../data/processed/ngrams/'+extension+'/ngrams_'+extension+'_'+language+'.csv', header=0)

		
		for entry in raw_ngrams['charTrigrams']:
			ngram_data[language]['charTrigrams'].add(str(entry))
		for entry in raw_ngrams['wordBigrams']:
			ngram_data[language]['wordBigrams'].add(str(entry))
		for entry in raw_ngrams['wordUnigrams']:
			ngram_data[language]['wordUnigrams'].add(str(entry))

	for fam in family:
		raw_ngrams = []
		raw_ngrams = pd.read_csv('../../data/processed/ngrams/'+extension+'/ngrams_'+extension+'_'+fam+'.csv', header=0)


		for entry in raw_ngrams['charTrigrams']:
			ngram_data[fam]['charTrigrams'].add(str(entry))
		for entry in raw_ngrams['wordBigrams']:
			ngram_data[fam]['wordBigrams'].add(str(entry))
		for entry in raw_ngrams['wordUnigrams']:
			ngram_data[fam]['wordUnigrams'].add(str(entry))

	for cat in category:
		raw_ngrams = []
		raw_ngrams = pd.read_csv('../../data/processed/ngrams/'+extension+'/ngrams_'+extension+'_'+cat+'.csv', header=0)


		for entry in raw_ngrams['charTrigrams']:
			ngram_data[cat]['charTrigrams'].add(str(entry))
		for entry in raw_ngrams['wordBigrams']:
			ngram_data[cat]['wordBigrams'].add(str(entry))
		for entry in raw_ngrams['wordUnigrams']:
			ngram_data[cat]['wordUnigrams'].add(str(entry))

	grouped = data[ngrams].groupby('lang',as_index=False)
	for x in grouped.groups:
		current = grouped.get_group(x)
		print('Current language: '+x)
		for index, row in current.iterrows():
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

			similarity = {	
						'French':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'German':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Greek':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'English':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Indian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Japanese':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Russian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}, 
						'Turkish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Bulgarian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Croatian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Czech':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Lithuanian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Polish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Serbian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Slovene':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Finnish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Dutch':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Norwegian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Swedish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Italian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Spanish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Portugese':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Romanian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Estonian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Hungarian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},

						'Balto-Slavic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Germanic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Romance':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Japonic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Turkic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Uralic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Indo-Aryan':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},

						'European':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'NonEuropean':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'ArtCul':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'BuiTecSci':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'Pol':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0},
						'SocSoc':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0}
						}

			for language in lang:
				similarity[language] = getNGramSimilarity(ngram_current, ngram_data[language])
				data.loc[index, 'charTrigrams_similarity_'+str(language)] = similarity[language]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(language)] = similarity[language]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(language)] = similarity[language]['wordUnigrams']
			for fam in family:
				similarity[fam] = getNGramSimilarity(ngram_current, ngram_data[fam])
				data.loc[index, 'charTrigrams_similarity_'+str(fam)] = similarity[fam]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(fam)] = similarity[fam]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(fam)] = similarity[fam]['wordUnigrams']
			for cat in category:
				similarity[cat] = getNGramSimilarity(ngram_current, ngram_data[cat])
				data.loc[index, 'charTrigrams_similarity_'+str(cat)] = similarity[cat]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(cat)] = similarity[cat]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(cat)] = similarity[cat]['wordUnigrams']

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

def tfidfScore(file,filters):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user', 'category']
	data = data[data.filteredSentence.str.contains('filteredSentence') == False]

	features_tfidf = [		'tfidf_French',
							'tfidf_German',
							'tfidf_Greek',
							'tfidf_Indian',
							'tfidf_Russian',
							'tfidf_Japanese',
							'tfidf_Turkish',
							'tfidf_Bulgarian',
							'tfidf_Croatian',
							'tfidf_Czech',
							'tfidf_Lithuanian',
							'tfidf_Polish',
							'tfidf_Serbian',
							'tfidf_Slovene',
							'tfidf_Finnish',
							'tfidf_Dutch',
							'tfidf_Norwegian',
							'tfidf_Swedish',
							'tfidf_Italian',
							'tfidf_Spanish',
							'tfidf_Portugese',
							'tfidf_Romanian',
							'tfidf_Estonian',
							'tfidf_Hungarian',
							'tfidf_English',
							'tfidf_Balto-Slavic',
							'tfidf_Germanic',
							'tfidf_Romance',
							'tfidf_Japonic',
							'tfidf_Turkic',
							'tfidf_Uralic',
							'tfidf_Indo-Aryan',
							'tfidf_European',
							'tfidf_NonEuropean',
							'tfidf_ArtCul',
							'tfidf_BuiTecSci',
							'tfidf_Pol',
							'tfidf_SocSoc'
							]

	for feature in features_tfidf:
		data[feature] = 0


	
	for index, entry in data['filteredSentence'].iteritems():

		grouped = data.groupby('lang',as_index=False)
		for x in grouped.groups:
			current = grouped.get_group(x)
			current.reset_index(drop=True,inplace=True)
			tfidflist = current['filteredSentence']
			tfidflist =  tfidflist.append(pd.DataFrame({entry}), ignore_index=True)

			tfidf_vectorizer=TfidfVectorizer(use_idf=True)
			tfidf_matrix=tfidf_vectorizer.fit_transform(tfidflist[0].values.tolist())
			data.loc[index, 'tfidf_'+x] = tfidf_matrix[-1].T.todense().max()

		grouped = data.groupby('langFam',as_index=False)
		for x in grouped.groups:
			current = grouped.get_group(x)
			current.reset_index(drop=True,inplace=True)
			tfidflist = current['filteredSentence']
			tfidflist =  tfidflist.append(pd.DataFrame({entry}), ignore_index=True)

			tfidf_vectorizer=TfidfVectorizer(use_idf=True)
			tfidf_matrix=tfidf_vectorizer.fit_transform(tfidflist[0].values.tolist())
			data.loc[index, 'tfidf_'+x] = tfidf_matrix[-1].T.todense().max()

		grouped = data.groupby('category',as_index=False)
		for x in grouped.groups:
			current = grouped.get_group(x)
			current.reset_index(drop=True,inplace=True)
			tfidflist = current['filteredSentence']
			tfidflist =  tfidflist.append(pd.DataFrame({entry}), ignore_index=True)

			tfidf_vectorizer=TfidfVectorizer(use_idf=True)
			tfidf_matrix=tfidf_vectorizer.fit_transform(tfidflist[0].values.tolist())
			data.loc[index, 'tfidf_'+x] = tfidf_matrix[-1].T.todense().max()

		if((index % 100 == 0)):
			print(index)



	


	


	## TFIDF
	# For each entry:
	# Calculate tfidf compared to each language, category and family
	# Save score in file
	data.to_csv("output/tfidf_"+filters+".csv", index=False)

if __name__ == "__main__":
	file = sys.argv[1]
	filters = 'none'
	arg2 = 'none'
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
		print('Splitting '+ file + ' with size: ' +filters + ' for '+arg2)
		splitFile(file,filters, arg2)
	elif(func == 'ngram'):
		print('Creating ngram model '+ file + ' with limit: ' +filters + ' for '+arg2)
		ngramModel(file,filters,arg2)
	elif(func == 'classifier'):
		print('Creating classification file for ' +filters)
		createClassifierFile(file,filters)
	elif(func == 'tfidf'):
		print('Adding tfidf scores for '+filters)
		tfidfScore(file,filters)
	print('Done')
	#print(result)