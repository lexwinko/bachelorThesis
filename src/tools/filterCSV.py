import sys
import pandas as pd
import numpy as np
import csv
import os
import re
from ngram import NGram
from nltk.util import ngrams
from nltk import WordPunctTokenizer
from collections import OrderedDict
import itertools
from gensim.models import Word2Vec
from sklearn.cluster import AffinityPropagation

from sklearn.feature_extraction.text import TfidfVectorizer 

file = ""
filters = ""

def reformatCSV(file, source, lang):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Output = pd.DataFrame()
	if(source == 'reddit'):
		txt_Import.columns = ['user', 'subreddit', 'post']
		txt_Output['text'] = txt_Import['post']
	else:
		txt_Import.columns = ['text', 'url', 'lang']
		txt_Output['text'] = txt_Import['text']
	txt_Output['lang'] = ''
	txt_Output['langFam'] = ''
	txt_Output['category'] = ''
	txt_Output['origin'] = ''

	
	

	filename = "reformat_"+source+"_"+lang+".csv"
	txt_Output.to_csv(filename, index=False)




def tagCSV(file, family, lang, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = ['text', 'lang', 'langFam', 'category', 'origin']

	txt_Import['langFam'] = family
	txt_Import['lang'] = lang
	txt_Import['origin'] = origin

	txt_Import.to_csv('tagged_'+file.split('.')[0]+'.csv',index=False)



def splitFile(file, ratio, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	if(origin == 'reddit'):
		txt_Import.columns = ['user', 'subreddit', 'post', 'langFam', 'lang', 'origin']
		txt_Import = txt_Import[txt_Import.subreddit.str.contains('subreddit') == False]
	else:
		txt_Import.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M', 'langFam', 'lang', 'user', 'category', 'origin']
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
		fileCol = ['user', 'subreddit', 'post', 'langFam', 'lang', 'origin']
	else:
		fileCol = [ 'correctedSentence','originalSentence','filteredSentence','stemmedSentence','elongated','caps','sentenceLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','hashtag','url','atUser','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user', 'category', 'origin']

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

def word2VecModel(file, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = [ 'correctedSentence','originalSentence','filteredSentence','stemmedSentence','elongated','caps','textLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','POSBigrams','functionWords','hashtag','url','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'category', 'origin']
	txt_Import = txt_Import[txt_Import.filteredSentence.str.contains('filteredSentence') == False]
	used_features =['correctedSentence', 'lang', 'langFam', 'category', 'origin']
	txt_Import = txt_Import[used_features]

	wpt = WordPunctTokenizer()
	tokenized_corpus = []
	for x in txt_Import['correctedSentence']:
		print(x)
		tokenized_corpus.append(wpt.tokenize(x.lower()))

	# Set values for various parameters
	feature_size = 10    # Word vector dimensionality  
	window_context = 10          # Context window size                                                                                    
	min_word_count = 1   # Minimum word count                        
	sample = 1e-3   # Downsample setting for frequent words

	w2v_model = Word2Vec(tokenized_corpus, size=feature_size, 
								  window=window_context, min_count = min_word_count,
								  sample=sample, iter=100)

	def average_word_vectors(words, model, vocabulary, num_features):
	
		feature_vector = np.zeros((num_features,),dtype="float64")
		nwords = 0.
		
		for word in words:
			if word in vocabulary: 
				nwords = nwords + 1.
				feature_vector = np.add(feature_vector, model[word])
		
		if nwords:
			feature_vector = np.divide(feature_vector, nwords)
			
		return feature_vector
	
   
	def averaged_word_vectorizer(corpus, model, num_features):
		vocabulary = set(model.wv.index2word)
		features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
						for tokenized_sentence in corpus]
		return np.array(features)


	# get document level embeddings
	#w2v_feature_array = averaged_word_vectorizer(corpus=tokenized_corpus, model=w2v_model, num_features=feature_size)

	similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
			for search_term in ['thank', 'number', 'too']}
	print(similar_words)
	# ap = AffinityPropagation()
	# ap.fit(w2v_feature_array)
	# cluster_labels = ap.labels_
	# cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
	# corpus_df = pd.concat([txt_Import['filteredSentence'], cluster_labels], axis=1)
	# print(corpus_df)



def ngramModel(file, origin):
	txt_Import = pd.read_csv(file, header=None, sep=',', skiprows=1)
	txt_Import.columns = [ 'correctedSentence','originalSentence','filteredSentence','stemmedSentence','elongated','caps','textLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','POSBigrams','functionWords','hashtag','url','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'category', 'origin']
	txt_Import = txt_Import[txt_Import.filteredSentence.str.contains('filteredSentence') == False]
	used_features =['charTrigrams', 'wordBigrams', 'wordUnigrams', 'POSBigrams', 'functionWords', 'lang', 'langFam', 'category', 'origin']
	txt_Import = txt_Import[used_features]

	limit = 1000

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
			'German':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'French':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Indian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Japanese':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Russian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Turkish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Bulgarian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Croatian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Czech':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Lithuanian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Polish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Serbian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Slovene':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Finnish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Dutch':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Norwegian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Swedish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Italian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Spanish':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Portuguese':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Romanian':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'English':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}
			}
	for x in grouped.groups:
		current = grouped.get_group(x)
		current.reset_index(drop=True,inplace=True)
		
		for charList in current['charTrigrams']:
			charList = charList.split(',')
			charListFilter = []
			for entry in range(0, len(charList)):
				charList[entry] = re.sub(r" '", r'', charList[entry])
				charList[entry] = re.sub(r"\[", r'', charList[entry])
				charList[entry] = re.sub(r"\]", r'', charList[entry])
				charList[entry] = re.sub(r"'", r'', charList[entry])
				charList[entry] = charList[entry].lower()
				charTest = charList[entry].split()
				if(len(charTest) > 0 and len(charTest[0]) == 3):
					charListFilter.append(charList[entry].split()[0])
			for chars in range(0,len(charListFilter)):
				if not charListFilter[chars] in ngrams[current['lang'].values[0]]['charTrigrams']:
					ngrams[current['lang'].values[0]]['charTrigrams'][charListFilter[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['charTrigrams'][charListFilter[chars]] += 1

		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['lang'].values[0]]['wordBigrams']:
					ngrams[current['lang'].values[0]]['wordBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['wordBigrams'][wordListFilter[chars]] += 1

		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['lang'].values[0]]['wordUnigrams']:
					ngrams[current['lang'].values[0]]['wordUnigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['wordUnigrams'][wordListFilter[chars]] += 1

		for wordList in current['POSBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['lang'].values[0]]['POSBigrams']:
					ngrams[current['lang'].values[0]]['POSBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['POSBigrams'][wordListFilter[chars]] += 1

		for wordList in current['functionWords']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['lang'].values[0]]['functionWords']:
					ngrams[current['lang'].values[0]]['functionWords'][wordListFilter[chars]] = 1
				else:
					ngrams[current['lang'].values[0]]['functionWords'][wordListFilter[chars]] += 1

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
					('Portuguese',{}),
					('Romanian',{}),
					('English',{})
					])
	for lang in ngrams:
		ngrams_descending[lang]['charTrigrams'] = sorted(ngrams[lang]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordBigrams'] = sorted(ngrams[lang]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['wordUnigrams'] = sorted(ngrams[lang]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['POSBigrams'] = sorted(ngrams[lang]['POSBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[lang]['functionWords'] = sorted(ngrams[lang]['functionWords'].items(), key=lambda t: t[1], reverse=True)

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
					('Portuguese',{}),
					('Romanian',{}),
					('English',{})
					])
	for lang in ngrams:
		ngrams_limited[lang]['charTrigrams'] = list(ngrams_descending[lang]['charTrigrams'])[:max(1,1000)]
		ngrams_limited[lang]['wordBigrams'] = list(ngrams_descending[lang]['wordBigrams'])[:max(1,300)]
		ngrams_limited[lang]['wordUnigrams'] = list(ngrams_descending[lang]['wordUnigrams'])[:max(1,500)]
		ngrams_limited[lang]['POSBigrams'] = list(ngrams_descending[lang]['POSBigrams'])[:max(1,300)]
		ngrams_limited[lang]['functionWords'] = list(ngrams_descending[lang]['functionWords'])[:max(1,300)]
	
	for lang in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+lang+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams', 'POSBigrams', 'functionWords']
			writer = csv.writer(f)
			writer.writerow(ngrams_limited[lang].keys())
			writer.writerows(itertools.zip_longest(*ngrams_limited[lang].values()))
			#writer = csv.DictWriter(f, fieldnames=fieldnames)
			#writer.writeheader()
			#for entry in range(0,min(len(ngrams_limited[lang]['charTrigrams']),len(ngrams_limited[lang]['wordBigrams']),len(ngrams_limited[lang]['wordUnigrams']),len(ngrams_limited[lang]['POSBigrams']),len(ngrams_limited[lang]['functionWords']),int(limit))):
			#for entry in range(0,int(limit)):
			#	if(entry < 300):
			#		writer.writerow({'charTrigrams': ngrams_limited[lang]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[lang]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[lang]['wordUnigrams'][entry][0] , 'POSBigrams': ngrams_limited[lang]['POSBigrams'][entry][0], 'functionWords': ngrams_limited[lang]['functionWords'][entry][0]})
			#	elif(entry >= 300 and entry < 500):
			#		writer.writerow({'charTrigrams': ngrams_limited[lang]['charTrigrams'][entry][0], 'wordBigrams': '' , 'wordUnigrams': ngrams_limited[lang]['wordUnigrams'][entry][0] , 'POSBigrams': '', 'functionWords': ''})


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
			'Native':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Turkic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Romance':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Greek':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Germanic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
			'Balto-Slavic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Japonic':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}},
			'Indo-Aryan':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}
			}
	for x in grouped_family.groups:
		current = grouped_family.get_group(x)
		current.reset_index(drop=True,inplace=True)
		
		for charList in current['charTrigrams']:
			charList = charList.split(',')
			charListFilter = []
			for entry in range(0, len(charList)):
				charList[entry] = re.sub(r" '", r'', charList[entry])
				charList[entry] = re.sub(r"\[", r'', charList[entry])
				charList[entry] = re.sub(r"\]", r'', charList[entry])
				charList[entry] = re.sub(r"'", r'', charList[entry])
				charList[entry] = charList[entry].lower()
				charTest = charList[entry].split()
				if(len(charTest) > 0 and len(charTest[0]) == 3):
					charListFilter.append(charList[entry].split()[0])
			for chars in range(0,len(charListFilter)):
				if not charListFilter[chars] in ngrams[current['langFam'].values[0]]['charTrigrams']:
					ngrams[current['langFam'].values[0]]['charTrigrams'][charListFilter[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['charTrigrams'][charListFilter[chars]] += 1

		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['langFam'].values[0]]['wordBigrams']:
					ngrams[current['langFam'].values[0]]['wordBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['wordBigrams'][wordListFilter[chars]] += 1

		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['langFam'].values[0]]['wordUnigrams']:
					ngrams[current['langFam'].values[0]]['wordUnigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['wordUnigrams'][wordListFilter[chars]] += 1

		for wordList in current['POSBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['langFam'].values[0]]['POSBigrams']:
					ngrams[current['langFam'].values[0]]['POSBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['POSBigrams'][wordListFilter[chars]] += 1

		for wordList in current['functionWords']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['langFam'].values[0]]['functionWords']:
					ngrams[current['langFam'].values[0]]['functionWords'][wordListFilter[chars]] = 1
				else:
					ngrams[current['langFam'].values[0]]['functionWords'][wordListFilter[chars]] += 1
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
					('Native',{}),
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
		ngrams_descending[family]['POSBigrams'] = sorted(ngrams[family]['POSBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[family]['functionWords'] = sorted(ngrams[family]['functionWords'].items(), key=lambda t: t[1], reverse=True)

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
					('Native',{}),
					('Turkic',{}),
					('Romance',{}),
					('Greek',{}),
					('Germanic',{}),
					('Balto-Slavic',{}),
					('Japonic',{}),
					('Indo-Aryan',{})
					])
	for family in ngrams:
		ngrams_limited[family]['charTrigrams'] = list(ngrams_descending[family]['charTrigrams'])[:max(1,1000)]
		ngrams_limited[family]['wordBigrams'] = list(ngrams_descending[family]['wordBigrams'])[:max(1,300)]
		ngrams_limited[family]['wordUnigrams'] = list(ngrams_descending[family]['wordUnigrams'])[:max(1,500)]
		ngrams_limited[family]['POSBigrams'] = list(ngrams_descending[family]['POSBigrams'])[:max(1,300)]
		ngrams_limited[family]['functionWords'] = list(ngrams_descending[family]['functionWords'])[:max(1,300)]
	
	for family in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+family+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams', 'POSBigrams', 'functionWords']
			writer = csv.writer(f)
			writer.writerow(ngrams_limited[family].keys())
			writer.writerows(itertools.zip_longest(*ngrams_limited[family].values()))
			#writer = csv.DictWriter(f, fieldnames=fieldnames)
			#writer.writeheader()
			#for entry in range(0,min(len(ngrams_limited[family]['charTrigrams']),len(ngrams_limited[family]['wordBigrams']),len(ngrams_limited[family]['wordUnigrams']),len(ngrams_limited[family]['POSBigrams']),len(ngrams_limited[family]['functionWords']),int(limit))):
		#		writer.writerow({'charTrigrams': ngrams_limited[family]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[family]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[family]['wordUnigrams'][entry][0] , 'POSBigrams': ngrams_limited[family]['POSBigrams'][entry][0] , 'functionWords': ngrams_limited[family]['functionWords'][entry][0]})

	grouped_category = txt_Import.groupby('category',as_index=False)
	ngrams = {
		'Reddit':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
		'Twitter':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}
		}
	for x in grouped_category.groups:
		current = grouped_category.get_group(x)
		current.reset_index(drop=True,inplace=True)
		
		for charList in current['charTrigrams']:
			charList = charList.split(',')
			charListFilter = []
			for entry in range(0, len(charList)):
				charList[entry] = re.sub(r" '", r'', charList[entry])
				charList[entry] = re.sub(r"\[", r'', charList[entry])
				charList[entry] = re.sub(r"\]", r'', charList[entry])
				charList[entry] = re.sub(r"'", r'', charList[entry])
				charList[entry] = charList[entry].lower()
				charTest = charList[entry].split()
				if(len(charTest) > 0 and len(charTest[0]) == 3):
					charListFilter.append(charList[entry].split()[0])
			for chars in range(0,len(charListFilter)):
				if not charListFilter[chars] in ngrams[current['category'].values[0]]['charTrigrams']:
					ngrams[current['category'].values[0]]['charTrigrams'][charListFilter[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['charTrigrams'][charListFilter[chars]] += 1

		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['category'].values[0]]['wordBigrams']:
					ngrams[current['category'].values[0]]['wordBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['wordBigrams'][wordListFilter[chars]] += 1

		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['category'].values[0]]['wordUnigrams']:
					ngrams[current['category'].values[0]]['wordUnigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['wordUnigrams'][wordListFilter[chars]] += 1

		for wordList in current['POSBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['category'].values[0]]['POSBigrams']:
					ngrams[current['category'].values[0]]['POSBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['POSBigrams'][wordListFilter[chars]] += 1

		for wordList in current['functionWords']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['category'].values[0]]['functionWords']:
					ngrams[current['category'].values[0]]['functionWords'][wordListFilter[chars]] = 1
				else:
					ngrams[current['category'].values[0]]['functionWords'][wordListFilter[chars]] += 1

	ngrams_descending = OrderedDict([
					('Reddit',{}),
					('Twitter',{})
					])
	for category in ngrams:
		ngrams_descending[category]['charTrigrams'] = sorted(ngrams[category]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[category]['wordBigrams'] = sorted(ngrams[category]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[category]['wordUnigrams'] = sorted(ngrams[category]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[category]['POSBigrams'] = sorted(ngrams[category]['POSBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[category]['functionWords'] = sorted(ngrams[category]['functionWords'].items(), key=lambda t: t[1], reverse=True)


	ngrams_limited = OrderedDict([
					('Reddit',{}),
					('Twitter',{})
					])
	for category in ngrams:
		ngrams_limited[category]['charTrigrams'] = list(ngrams_descending[category]['charTrigrams'])[:max(1,1000)]
		ngrams_limited[category]['wordBigrams'] = list(ngrams_descending[category]['wordBigrams'])[:max(1,300)]
		ngrams_limited[category]['wordUnigrams'] = list(ngrams_descending[category]['wordUnigrams'])[:max(1,500)]
		ngrams_limited[category]['POSBigrams'] = list(ngrams_descending[category]['POSBigrams'])[:max(1,300)]
		ngrams_limited[category]['functionWords'] = list(ngrams_descending[category]['functionWords'])[:max(1,300)]
	
	for category in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+category+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams', 'POSBigrams', 'functionWords']
			writer = csv.writer(f)
			writer.writerow(ngrams_limited[category].keys())
			writer.writerows(itertools.zip_longest(*ngrams_limited[category].values()))
			#writer = csv.DictWriter(f, fieldnames=fieldnames)
			#writer.writeheader()
			#for entry in range(0,min(len(ngrams_limited[category]['charTrigrams']),len(ngrams_limited[category]['wordBigrams']),len(ngrams_limited[category]['wordUnigrams']),len(ngrams_limited[category]['POSBigrams']),len(ngrams_limited[category]['functionWords']),int(limit))):
		#		writer.writerow({'charTrigrams': ngrams_limited[category]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[category]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[category]['wordUnigrams'][entry][0] , 'POSBigrams': ngrams_limited[category]['POSBigrams'][entry][0] , 'functionWords': ngrams_limited[category]['functionWords'][entry][0]})

	grouped_origin = txt_Import.groupby('origin',as_index=False)
	ngrams = {
		'Native':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}, 
		'NonNative':{'charTrigrams':{},'wordBigrams':{},'wordUnigrams':{},'POSBigrams':{},'functionWords':{}}
		}
	for x in grouped_origin.groups:
		current = grouped_origin.get_group(x)
		current.reset_index(drop=True,inplace=True)
		
		for charList in current['charTrigrams']:
			charList = charList.split(',')
			charListFilter = []
			for entry in range(0, len(charList)):
				charList[entry] = re.sub(r" '", r'', charList[entry])
				charList[entry] = re.sub(r"\[", r'', charList[entry])
				charList[entry] = re.sub(r"\]", r'', charList[entry])
				charList[entry] = re.sub(r"'", r'', charList[entry])
				charList[entry] = charList[entry].lower()
				charTest = charList[entry].split()
				if(len(charTest) > 0 and len(charTest[0]) == 3):
					charListFilter.append(charList[entry].split()[0])
			for chars in range(0,len(charListFilter)):
				if not charListFilter[chars] in ngrams[current['origin'].values[0]]['charTrigrams']:
					ngrams[current['origin'].values[0]]['charTrigrams'][charListFilter[chars]] = 1
				else:
					ngrams[current['origin'].values[0]]['charTrigrams'][charListFilter[chars]] += 1

		for wordList in current['wordBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['origin'].values[0]]['wordBigrams']:
					ngrams[current['origin'].values[0]]['wordBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['origin'].values[0]]['wordBigrams'][wordListFilter[chars]] += 1

		for wordList in current['wordUnigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['origin'].values[0]]['wordUnigrams']:
					ngrams[current['origin'].values[0]]['wordUnigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['origin'].values[0]]['wordUnigrams'][wordListFilter[chars]] += 1

		for wordList in current['POSBigrams']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 1):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['origin'].values[0]]['POSBigrams']:
					ngrams[current['origin'].values[0]]['POSBigrams'][wordListFilter[chars]] = 1
				else:
					ngrams[current['origin'].values[0]]['POSBigrams'][wordListFilter[chars]] += 1

		for wordList in current['functionWords']:
			wordList = wordList.split(',')
			wordListFilter = []
			for entry in range(0, len(wordList)):
				wordList[entry] = re.sub(r" '", r'', wordList[entry])
				wordList[entry] = re.sub(r"\[", r'', wordList[entry])
				wordList[entry] = re.sub(r"\]", r'', wordList[entry])
				wordList[entry] = re.sub(r"'", r'', wordList[entry])
				wordList[entry] = wordList[entry].lower()
				wordTest = wordList[entry].split()
				if(len(wordTest) > 0):
					wordListFilter.append(wordList[entry])
			for chars in range(0,len(wordListFilter)):
				if not wordListFilter[chars] in ngrams[current['origin'].values[0]]['functionWords']:
					ngrams[current['origin'].values[0]]['functionWords'][wordListFilter[chars]] = 1
				else:
					ngrams[current['origin'].values[0]]['functionWords'][wordListFilter[chars]] += 1

	ngrams_descending = OrderedDict([
					('Native',{}),
					('NonNative',{})
					])
	for origins in ngrams:
		ngrams_descending[origins]['charTrigrams'] = sorted(ngrams[origins]['charTrigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[origins]['wordBigrams'] = sorted(ngrams[origins]['wordBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[origins]['wordUnigrams'] = sorted(ngrams[origins]['wordUnigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[origins]['POSBigrams'] = sorted(ngrams[origins]['POSBigrams'].items(), key=lambda t: t[1], reverse=True)
		ngrams_descending[origins]['functionWords'] = sorted(ngrams[origins]['functionWords'].items(), key=lambda t: t[1], reverse=True)


	ngrams_limited = OrderedDict([
					('Native',{}),
					('NonNative',{})
					])
	for origins in ngrams:
		ngrams_limited[origins]['charTrigrams'] = list(ngrams_descending[origins]['charTrigrams'])[:max(1,1000)]
		ngrams_limited[origins]['wordBigrams'] = list(ngrams_descending[origins]['wordBigrams'])[:max(1,300)]
		ngrams_limited[origins]['wordUnigrams'] = list(ngrams_descending[origins]['wordUnigrams'])[:max(1,500)]
		ngrams_limited[origins]['POSBigrams'] = list(ngrams_descending[origins]['POSBigrams'])[:max(1,300)]
		ngrams_limited[origins]['functionWords'] = list(ngrams_descending[origins]['functionWords'])[:max(1,300)]
	
	for origins in ngrams_limited:
		filename = "output/ngrams_"+origin+"_"+origins+".csv"
		with open(filename, "w") as f:
			fieldnames = ['charTrigrams', 'wordBigrams', 'wordUnigrams', 'POSBigrams', 'functionWords']
			writer = csv.writer(f)
			writer.writerow(ngrams_limited[origins].keys())
			writer.writerows(itertools.zip_longest(*ngrams_limited[origins].values()))
			#writer = csv.DictWriter(f, fieldnames=fieldnames)
			#writer.writeheader()
			#for entry in range(0,min(len(ngrams_limited[origins]['charTrigrams']),len(ngrams_limited[origins]['wordBigrams']),len(ngrams_limited[origins]['wordUnigrams']),len(ngrams_limited[origins]['POSBigrams']),len(ngrams_limited[origins]['functionWords']),int(limit))):
			#	writer.writerow({'charTrigrams': ngrams_limited[origins]['charTrigrams'][entry][0], 'wordBigrams': ngrams_limited[origins]['wordBigrams'][entry][0] , 'wordUnigrams': ngrams_limited[origins]['wordUnigrams'][entry][0] , 'POSBigrams': ngrams_limited[origins]['POSBigrams'][entry][0] , 'functionWords': ngrams_limited[origins]['functionWords'][entry][0]})


def createClassifierFile(file,filters):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	data.columns = [ 'correctedSentence','originalSentence','filteredSentence','stemmedSentence','elongated','caps','textLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','POSBigrams','functionWords','hashtag','url','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'category', 'origin']
	data = data[data.filteredSentence.str.contains('filteredSentence') == False]

	if(filters == 'redditE'):
		extension = 'reddit'
		version = 'european'
	elif(filters == 'redditNE'):
		extension = 'reddit'
		version = 'noneuropean'
	elif(filters == 'combinedE'):
		extension = 'combined'
		version = 'european'
	elif(filters == 'combinedNE'):
		extension = 'combined'
		version = 'noneuropean'
	elif(filters == 'twitter'):
		extension = 'twitter'
	# 	category= [
	# 			'ArtCul',
	# 			'BuiTecSci',
	# 			'Pol',
	# 			'SocSoc'
	# 			]
	# else:
	# 	extension = 'combined'
	# 	category = [
	# 			'ArtCul',
	# 			'BuiTecSci',
	# 			'Pol',
	# 			'SocSoc',
	# 			'European',
	# 			'NonEuropean'
	# 	]

	if(extension == 'combined'):

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
				'Portuguese',
				'Romanian'
				]
		family= [
				'Balto-Slavic',
				'Germanic',
				'Indo-Aryan',
				'Japonic',
				'Romance',
				'Turkic',
				'Greek',
				'Native'
				]

		category = [
				'Reddit',
				'Twitter']

		origin = [
				'Native',
				'NonNative']
	elif(extension == 'reddit'):
		lang = [
				'French', 
				'German', 
				'English', 
				'Russian', 
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
				'Portuguese',
				'Romanian'
				]
		family= [
				'Balto-Slavic',
				'Germanic',
				'Romance',
				'Native'
				]

		category = [
				'Reddit']

		origin = [
				'Native',
				'NonNative']

	elif(extension == 'twitter'):
		lang = [
				'French', 
				'German', 
				'Greek', 
				'English', 
				'Indian', 
				'Japanese', 
				'Russian', 
				'Turkish'
				]
		family= [
				'Balto-Slavic',
				'Germanic',
				'Indo-Aryan',
				'Japonic',
				'Romance',
				'Turkic',
				'Greek',
				'Native'
				]

		category = [
				'Twitter']

		origin = [
				'Native',
				'NonNative']

	ngrams = ['charTrigrams','wordBigrams','wordUnigrams', 'POSBigrams', 'functionWords', 'lang', 'langFam', 'category', 'origin']
	features_similarity = [	'charTrigrams_similarity_French',
							'wordBigrams_similarity_French',
							'wordUnigrams_similarity_French',
							'POSBigrams_similarity_French',
							'functionWords_similarity_French',

							'charTrigrams_similarity_German',
							'wordBigrams_similarity_German',
							'wordUnigrams_similarity_German',
							'POSBigrams_similarity_German',
							'functionWords_similarity_German',

							'charTrigrams_similarity_Greek',
							'wordBigrams_similarity_Greek',
							'wordUnigrams_similarity_Greek',
							'POSBigrams_similarity_Greek',
							'functionWords_similarity_Greek',

							'charTrigrams_similarity_Indian',
							'wordBigrams_similarity_Indian',
							'wordUnigrams_similarity_Indian',
							'POSBigrams_similarity_Indian',
							'functionWords_similarity_Indian',

							'charTrigrams_similarity_Russian',
							'wordBigrams_similarity_Russian',
							'wordUnigrams_similarity_Russian',
							'POSBigrams_similarity_Russian',
							'functionWords_similarity_Russian',

							'charTrigrams_similarity_Japanese',
							'wordBigrams_similarity_Japanese',
							'wordUnigrams_similarity_Japanese',
							'POSBigrams_similarity_Japanese',
							'functionWords_similarity_Japanese',

							'charTrigrams_similarity_Turkish',
							'wordBigrams_similarity_Turkish',
							'wordUnigrams_similarity_Turkish',
							'POSBigrams_similarity_Turkish',
							'functionWords_similarity_Turkish',

							'charTrigrams_similarity_Bulgarian',
							'wordBigrams_similarity_Bulgarian',
							'wordUnigrams_similarity_Bulgarian',
							'POSBigrams_similarity_Bulgarian',
							'functionWords_similarity_Bulgarian',

							'charTrigrams_similarity_Croatian',
							'wordBigrams_similarity_Croatian',
							'wordUnigrams_similarity_Croatian',
							'POSBigrams_similarity_Croatian',
							'functionWords_similarity_Croatian',

							'charTrigrams_similarity_Czech',
							'wordBigrams_similarity_Czech',
							'wordUnigrams_similarity_Czech',
							'POSBigrams_similarity_Czech',
							'functionWords_similarity_Czech',

							'charTrigrams_similarity_Lithuanian',
							'wordBigrams_similarity_Lithuanian',
							'wordUnigrams_similarity_Lithuanian',
							'POSBigrams_similarity_Lithuanian',
							'functionWords_similarity_Lithuanian',

							'charTrigrams_similarity_Polish',
							'wordBigrams_similarity_Polish',
							'wordUnigrams_similarity_Polish',
							'POSBigrams_similarity_Polish',
							'functionWords_similarity_Polish',

							'charTrigrams_similarity_Serbian',
							'wordBigrams_similarity_Serbian',
							'wordUnigrams_similarity_Serbian',
							'POSBigrams_similarity_Serbian',
							'functionWords_similarity_Serbian',

							'charTrigrams_similarity_Slovene',
							'wordBigrams_similarity_Slovene',
							'wordUnigrams_similarity_Slovene',
							'POSBigrams_similarity_Slovene',
							'functionWords_similarity_Slovene',

							'charTrigrams_similarity_Finnish',
							'wordBigrams_similarity_Finnish',
							'wordUnigrams_similarity_Finnish',
							'POSBigrams_similarity_Finnish',
							'functionWords_similarity_Finnish',

							'charTrigrams_similarity_Dutch',
							'wordBigrams_similarity_Dutch',
							'wordUnigrams_similarity_Dutch',
							'POSBigrams_similarity_Dutch',
							'functionWords_similarity_Dutch',

							'charTrigrams_similarity_Norwegian',
							'wordBigrams_similarity_Norwegian',
							'wordUnigrams_similarity_Norwegian',
							'POSBigrams_similarity_Norwegian',
							'functionWords_similarity_Norwegian',

							'charTrigrams_similarity_Swedish',
							'wordBigrams_similarity_Swedish',
							'wordUnigrams_similarity_Swedish',
							'POSBigrams_similarity_Swedish',
							'functionWords_similarity_Swedish',

							'charTrigrams_similarity_Italian',
							'wordBigrams_similarity_Italian',
							'wordUnigrams_similarity_Italian',
							'POSBigrams_similarity_Italian',
							'functionWords_similarity_Italian',

							'charTrigrams_similarity_Spanish',
							'wordBigrams_similarity_Spanish',
							'wordUnigrams_similarity_Spanish',
							'POSBigrams_similarity_Spanish',
							'functionWords_similarity_Spanish',

							'charTrigrams_similarity_Portuguese',
							'wordBigrams_similarity_Portuguese',
							'wordUnigrams_similarity_Portuguese',
							'POSBigrams_similarity_Portuguese',
							'functionWords_similarity_Portuguese',

							'charTrigrams_similarity_Romanian',
							'wordBigrams_similarity_Romanian',
							'wordUnigrams_similarity_Romanian',
							'POSBigrams_similarity_Romanian',
							'functionWords_similarity_Romanian',

							'charTrigrams_similarity_Balto-Slavic',
							'wordBigrams_similarity_Balto-Slavic',
							'wordUnigrams_similarity_Balto-Slavic',
							'POSBigrams_similarity_Balto-Slavic',
							'functionWords_similarity_Balto-Slavic',

							'charTrigrams_similarity_Germanic',
							'wordBigrams_similarity_Germanic',
							'wordUnigrams_similarity_Germanic',
							'POSBigrams_similarity_Germanic',
							'functionWords_similarity_Germanic',

							'charTrigrams_similarity_Romance',
							'wordBigrams_similarity_Romance',
							'wordUnigrams_similarity_Romance',
							'POSBigrams_similarity_Romance',
							'functionWords_similarity_Romance',

							'charTrigrams_similarity_Japonic',
							'wordBigrams_similarity_Japonic',
							'wordUnigrams_similarity_Japonic',
							'POSBigrams_similarity_Japonic',
							'functionWords_similarity_Japonic',

							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English',
							'POSBigrams_similarity_English',
							'functionWords_similarity_English',

							'charTrigrams_similarity_Turkic',
							'wordBigrams_similarity_Turkic',
							'wordUnigrams_similarity_Turkic',
							'POSBigrams_similarity_Turkic',
							'functionWords_similarity_Turkic',

							'charTrigrams_similarity_Indo-Aryan',
							'wordBigrams_similarity_Indo-Aryan',
							'wordUnigrams_similarity_Indo-Aryan',
							'POSBigrams_similarity_Indo-Aryan',
							'functionWords_similarity_Indo-Aryan',

							'charTrigrams_similarity_Native',
							'wordBigrams_similarity_Native',
							'wordUnigrams_similarity_Native',
							'POSBigrams_similarity_Native',
							'functionWords_similarity_Native',

							'charTrigrams_similarity_NonNative',
							'wordBigrams_similarity_NonNative',
							'wordUnigrams_similarity_NonNative',
							'POSBigrams_similarity_NonNative',
							'functionWords_similarity_NonNative',

							'charTrigrams_similarity_Reddit',
							'wordBigrams_similarity_Reddit',
							'wordUnigrams_similarity_Reddit',
							'POSBigrams_similarity_Reddit',
							'functionWords_similarity_Reddit',

							'charTrigrams_similarity_Twitter',
							'wordBigrams_similarity_Twitter',
							'wordUnigrams_similarity_Twitter',
							'POSBigrams_similarity_Twitter',
							'functionWords_similarity_Twitter'
							]

	for feature in features_similarity:
		data[feature] = 0


	ngram_data = {	
					'French':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}, 
					'German':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}, 
					'Greek':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}, 
					'English':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Indian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}, 
					'Japanese':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}, 
					'Russian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}, 
					'Turkish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Bulgarian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Croatian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Czech':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Lithuanian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Polish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Serbian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Slovene':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Finnish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Dutch':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Norwegian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Swedish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Italian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Spanish':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Portuguese':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Romanian':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Balto-Slavic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Germanic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Romance':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Japonic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Turkic':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Indo-Aryan':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Native':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'NonNative':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Reddit':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()},
					'Twitter':{'charTrigrams':NGram(),'wordBigrams':NGram(),'wordUnigrams':NGram(),'POSBigrams':NGram(),'functionWords':NGram()}
					}
	for language in lang:
		print(language)
		raw_ngrams = []
		if not(filters == 'twitter'):
			raw_ngrams = pd.read_csv('../../data/processed/'+extension+'/ngrams/'+version+'/ngrams_'+extension+'_'+language+'.csv', header=0)
		else:
			raw_ngrams = pd.read_csv('../../data/processed/'+extension+'/ngrams/ngrams_'+extension+'_'+language+'.csv', header=0)

		print(raw_ngrams.count())
		for entry in raw_ngrams['charTrigrams'].head(raw_ngrams.count()[0]):
			ngram_data[language]['charTrigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordBigrams'].head(raw_ngrams.count()[1]):
			ngram_data[language]['wordBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordUnigrams'].head(raw_ngrams.count()[2]):
			ngram_data[language]['wordUnigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['POSBigrams'].head(raw_ngrams.count()[3]):
			ngram_data[language]['POSBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['functionWords'].head(raw_ngrams.count()[4]):
			ngram_data[language]['functionWords'].add(re.search(r"\'(.*?)\'", entry).group(1))

	for fam in family:
		raw_ngrams = []
		if not(filters == 'twitter'):
			raw_ngrams = pd.read_csv('../../data/processed/'+extension+'/ngrams/'+version+'/ngrams_'+extension+'_'+fam+'.csv', header=0)
		else:
			raw_ngrams = pd.read_csv('../../data/processed/'+extension+'/ngrams/ngrams_'+extension+'_'+fam+'.csv', header=0)


		for entry in raw_ngrams['charTrigrams'].head(raw_ngrams.count()[0]):
			ngram_data[fam]['charTrigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordBigrams'].head(raw_ngrams.count()[1]):
			ngram_data[fam]['wordBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordUnigrams'].head(raw_ngrams.count()[2]):
			ngram_data[fam]['wordUnigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['POSBigrams'].head(raw_ngrams.count()[3]):
			ngram_data[fam]['POSBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['functionWords'].head(raw_ngrams.count()[4]):
			ngram_data[fam]['functionWords'].add(re.search(r"\'(.*?)\'", entry).group(1))

	for cat in category:
		raw_ngrams = []
		if not(filters == 'twitter'):
			raw_ngrams = pd.read_csv('../../data/processed/'+extension+'/ngrams/'+version+'/ngrams_'+extension+'_'+cat+'.csv', header=0)
		else:
			raw_ngrams = pd.read_csv('../../data/processed/'+extension+'/ngrams/ngrams_'+extension+'_'+cat+'.csv', header=0)


		for entry in raw_ngrams['charTrigrams'].head(raw_ngrams.count()[0]):
			ngram_data[cat]['charTrigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordBigrams'].head(raw_ngrams.count()[1]):
			ngram_data[cat]['wordBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordUnigrams'].head(raw_ngrams.count()[2]):
			ngram_data[cat]['wordUnigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['POSBigrams'].head(raw_ngrams.count()[3]):
			ngram_data[cat]['POSBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['functionWords'].head(raw_ngrams.count()[4]):
			ngram_data[cat]['functionWords'].add(re.search(r"\'(.*?)\'", entry).group(1))

	for origins in origin:
		raw_ngrams = []
		if not(filters == 'twitter'):
			file = '../../data/processed/'+extension+'/ngrams/'+version+'/ngrams_'+extension+'_'+origins+'.csv'
		else:
			file = '../../data/processed/'+extension+'/ngrams/ngrams_'+extension+'_'+origins+'.csv'

		raw_ngrams = pd.read_csv(file, header=0)
		for entry in raw_ngrams['charTrigrams'].head(raw_ngrams.count()[0]):
			ngram_data[origins]['charTrigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordBigrams'].head(raw_ngrams.count()[1]):
			ngram_data[origins]['wordBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['wordUnigrams'].head(raw_ngrams.count()[2]):
			ngram_data[origins]['wordUnigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['POSBigrams'].head(raw_ngrams.count()[3]):
			ngram_data[origins]['POSBigrams'].add(re.search(r"\'(.*?)\'", entry).group(1))
		for entry in raw_ngrams['functionWords'].head(raw_ngrams.count()[4]):
			ngram_data[origins]['functionWords'].add(re.search(r"\'(.*?)\'", entry).group(1))


	grouped = data[ngrams].groupby('lang',as_index=False)
	for x in grouped.groups:
		current = grouped.get_group(x)
		print('Current language: '+x)
		for index, row in current.iterrows():
			ngram_current = {'charTrigrams':NGram(), 'wordBigrams':NGram(), 'wordUnigrams':NGram(), 'POSBigrams':NGram(), 'functionWords':NGram()}
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
			ngramlist = row['POSBigrams'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['POSBigrams'].add(str(ngramlist[entry]))
			ngramlist = row['functionWords'].split(',')
			for entry in range(0, len(ngramlist)):
				ngramlist[entry] = re.sub(r" '", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\[", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"\]", r'', ngramlist[entry])
				ngramlist[entry] = re.sub(r"'", r'', ngramlist[entry])
				ngramlist[entry] = ngramlist[entry].lower()
				ngram_current['functionWords'].add(str(ngramlist[entry]))

			similarity = {	
						'French':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}, 
						'German':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}, 
						'Greek':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}, 
						'English':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Indian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}, 
						'Japanese':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}, 
						'Russian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}, 
						'Turkish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Bulgarian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Croatian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Czech':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Lithuanian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Polish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Serbian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Slovene':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Finnish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Dutch':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Norwegian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Swedish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Italian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Spanish':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Portuguese':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Romanian':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},

						'Balto-Slavic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Germanic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Romance':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Japonic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Turkic':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Indo-Aryan':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},

						'Native':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'NonNative':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},

						'Reddit':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0},
						'Twitter':{'charTrigrams':0,'wordBigrams':0,'wordUnigrams':0,'POSBigrams':0,'functionWords':0}
						}

			for language in lang:
				similarity[language] = getNGramSimilarity(ngram_current, ngram_data[language])
				data.loc[index, 'charTrigrams_similarity_'+str(language)] = similarity[language]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(language)] = similarity[language]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(language)] = similarity[language]['wordUnigrams']
				data.loc[index, 'POSBigrams_similarity_'+str(language)] = similarity[language]['POSBigrams']
				data.loc[index, 'functionWords_similarity_'+str(language)] = similarity[language]['functionWords']
			for fam in family:
				similarity[fam] = getNGramSimilarity(ngram_current, ngram_data[fam])
				data.loc[index, 'charTrigrams_similarity_'+str(fam)] = similarity[fam]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(fam)] = similarity[fam]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(fam)] = similarity[fam]['wordUnigrams']
				data.loc[index, 'POSBigrams_similarity_'+str(fam)] = similarity[fam]['POSBigrams']
				data.loc[index, 'functionWords_similarity_'+str(fam)] = similarity[fam]['functionWords']
			for cat in category:
				similarity[cat] = getNGramSimilarity(ngram_current, ngram_data[cat])
				data.loc[index, 'charTrigrams_similarity_'+str(cat)] = similarity[cat]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(cat)] = similarity[cat]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(cat)] = similarity[cat]['wordUnigrams']
				data.loc[index, 'POSBigrams_similarity_'+str(cat)] = similarity[cat]['POSBigrams']
				data.loc[index, 'functionWords_similarity_'+str(cat)] = similarity[cat]['functionWords']
			for origins in origin:
				similarity[origins] = getNGramSimilarity(ngram_current, ngram_data[origins])
				data.loc[index, 'charTrigrams_similarity_'+str(origins)] = similarity[origins]['charTrigrams']
				data.loc[index, 'wordBigrams_similarity_'+str(origins)] = similarity[origins]['wordBigrams']
				data.loc[index, 'wordUnigrams_similarity_'+str(origins)] = similarity[origins]['wordUnigrams']
				data.loc[index, 'POSBigrams_similarity_'+str(origins)] = similarity[origins]['POSBigrams']
				data.loc[index, 'functionWords_similarity_'+str(origins)] = similarity[origins]['functionWords']

	data.to_csv("output/classification_data_"+filters+".csv", index=False)

def getNGramSimilarity(ngrams, data):
	#print("similarity")
	#print('ngrams', len(ngrams['charTrigrams']), len(ngrams['wordBigrams']))
	#print('data',len(data['charTrigrams']), len(data['wordBigrams']))
	intersection = {'charTrigrams': list(ngrams['charTrigrams'].intersection(data['charTrigrams'])), 'wordBigrams':list(ngrams['wordBigrams'].intersection(data['wordBigrams'])), 'wordUnigrams':list(ngrams['wordUnigrams'].intersection(data['wordUnigrams'])), 'POSBigrams':list(ngrams['POSBigrams'].intersection(data['POSBigrams'])), 'functionWords':list(ngrams['functionWords'].intersection(data['functionWords']))}
	union = {'charTrigrams': list(ngrams['charTrigrams'].union(data['charTrigrams'])), 'wordBigrams':list(ngrams['wordBigrams'].union(data['wordBigrams'])), 'wordUnigrams':list(ngrams['wordUnigrams'].union(data['wordUnigrams'])), 'POSBigrams':list(ngrams['POSBigrams'].union(data['POSBigrams'])), 'functionWords':list(ngrams['functionWords'].union(data['functionWords']))}
	#print(len(intersection['charTrigrams']), len(intersection['wordBigrams']), len(union['charTrigrams']), len(union['wordBigrams']))
	similariy = {'charTrigrams': NGram.ngram_similarity(len(intersection['charTrigrams']), len(union['charTrigrams'])), 'wordBigrams': NGram.ngram_similarity(len(intersection['wordBigrams']), len(union['wordBigrams'])), 'wordUnigrams': NGram.ngram_similarity(len(intersection['wordUnigrams']), len(union['wordUnigrams'])), 'POSBigrams': NGram.ngram_similarity(len(intersection['POSBigrams']), len(union['POSBigrams'])), 'functionWords': NGram.ngram_similarity(len(intersection['functionWords']), len(union['functionWords']))}
	return similariy

def calculateFeatureAverage(file, source, other):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence', 'elongated','caps','textLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'POSBigrams', 'functionWords', 'hashtag', 'url', '#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'category', 'origin',
						'charTrigrams_similarity_French',
							'wordBigrams_similarity_French',
							'wordUnigrams_similarity_French',
							'POSBigrams_similarity_French',
							'functionWords_similarity_French',

							'charTrigrams_similarity_German',
							'wordBigrams_similarity_German',
							'wordUnigrams_similarity_German',
							'POSBigrams_similarity_German',
							'functionWords_similarity_German',

							'charTrigrams_similarity_Greek',
							'wordBigrams_similarity_Greek',
							'wordUnigrams_similarity_Greek',
							'POSBigrams_similarity_Greek',
							'functionWords_similarity_Greek',

							'charTrigrams_similarity_Indian',
							'wordBigrams_similarity_Indian',
							'wordUnigrams_similarity_Indian',
							'POSBigrams_similarity_Indian',
							'functionWords_similarity_Indian',

							'charTrigrams_similarity_Russian',
							'wordBigrams_similarity_Russian',
							'wordUnigrams_similarity_Russian',
							'POSBigrams_similarity_Russian',
							'functionWords_similarity_Russian',

							'charTrigrams_similarity_Japanese',
							'wordBigrams_similarity_Japanese',
							'wordUnigrams_similarity_Japanese',
							'POSBigrams_similarity_Japanese',
							'functionWords_similarity_Japanese',

							'charTrigrams_similarity_Turkish',
							'wordBigrams_similarity_Turkish',
							'wordUnigrams_similarity_Turkish',
							'POSBigrams_similarity_Turkish',
							'functionWords_similarity_Turkish',

							'charTrigrams_similarity_Bulgarian',
							'wordBigrams_similarity_Bulgarian',
							'wordUnigrams_similarity_Bulgarian',
							'POSBigrams_similarity_Bulgarian',
							'functionWords_similarity_Bulgarian',

							'charTrigrams_similarity_Croatian',
							'wordBigrams_similarity_Croatian',
							'wordUnigrams_similarity_Croatian',
							'POSBigrams_similarity_Croatian',
							'functionWords_similarity_Croatian',

							'charTrigrams_similarity_Czech',
							'wordBigrams_similarity_Czech',
							'wordUnigrams_similarity_Czech',
							'POSBigrams_similarity_Czech',
							'functionWords_similarity_Czech',

							'charTrigrams_similarity_Lithuanian',
							'wordBigrams_similarity_Lithuanian',
							'wordUnigrams_similarity_Lithuanian',
							'POSBigrams_similarity_Lithuanian',
							'functionWords_similarity_Lithuanian',

							'charTrigrams_similarity_Polish',
							'wordBigrams_similarity_Polish',
							'wordUnigrams_similarity_Polish',
							'POSBigrams_similarity_Polish',
							'functionWords_similarity_Polish',

							'charTrigrams_similarity_Serbian',
							'wordBigrams_similarity_Serbian',
							'wordUnigrams_similarity_Serbian',
							'POSBigrams_similarity_Serbian',
							'functionWords_similarity_Serbian',

							'charTrigrams_similarity_Slovene',
							'wordBigrams_similarity_Slovene',
							'wordUnigrams_similarity_Slovene',
							'POSBigrams_similarity_Slovene',
							'functionWords_similarity_Slovene',

							'charTrigrams_similarity_Finnish',
							'wordBigrams_similarity_Finnish',
							'wordUnigrams_similarity_Finnish',
							'POSBigrams_similarity_Finnish',
							'functionWords_similarity_Finnish',

							'charTrigrams_similarity_Dutch',
							'wordBigrams_similarity_Dutch',
							'wordUnigrams_similarity_Dutch',
							'POSBigrams_similarity_Dutch',
							'functionWords_similarity_Dutch',

							'charTrigrams_similarity_Norwegian',
							'wordBigrams_similarity_Norwegian',
							'wordUnigrams_similarity_Norwegian',
							'POSBigrams_similarity_Norwegian',
							'functionWords_similarity_Norwegian',

							'charTrigrams_similarity_Swedish',
							'wordBigrams_similarity_Swedish',
							'wordUnigrams_similarity_Swedish',
							'POSBigrams_similarity_Swedish',
							'functionWords_similarity_Swedish',

							'charTrigrams_similarity_Italian',
							'wordBigrams_similarity_Italian',
							'wordUnigrams_similarity_Italian',
							'POSBigrams_similarity_Italian',
							'functionWords_similarity_Italian',

							'charTrigrams_similarity_Spanish',
							'wordBigrams_similarity_Spanish',
							'wordUnigrams_similarity_Spanish',
							'POSBigrams_similarity_Spanish',
							'functionWords_similarity_Spanish',

							'charTrigrams_similarity_Portuguese',
							'wordBigrams_similarity_Portuguese',
							'wordUnigrams_similarity_Portuguese',
							'POSBigrams_similarity_Portuguese',
							'functionWords_similarity_Portuguese',

							'charTrigrams_similarity_Romanian',
							'wordBigrams_similarity_Romanian',
							'wordUnigrams_similarity_Romanian',
							'POSBigrams_similarity_Romanian',
							'functionWords_similarity_Romanian',

							'charTrigrams_similarity_Balto-Slavic',
							'wordBigrams_similarity_Balto-Slavic',
							'wordUnigrams_similarity_Balto-Slavic',
							'POSBigrams_similarity_Balto-Slavic',
							'functionWords_similarity_Balto-Slavic',

							'charTrigrams_similarity_Germanic',
							'wordBigrams_similarity_Germanic',
							'wordUnigrams_similarity_Germanic',
							'POSBigrams_similarity_Germanic',
							'functionWords_similarity_Germanic',

							'charTrigrams_similarity_Romance',
							'wordBigrams_similarity_Romance',
							'wordUnigrams_similarity_Romance',
							'POSBigrams_similarity_Romance',
							'functionWords_similarity_Romance',

							'charTrigrams_similarity_Japonic',
							'wordBigrams_similarity_Japonic',
							'wordUnigrams_similarity_Japonic',
							'POSBigrams_similarity_Japonic',
							'functionWords_similarity_Japonic',

							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English',
							'POSBigrams_similarity_English',
							'functionWords_similarity_English',

							'charTrigrams_similarity_Turkic',
							'wordBigrams_similarity_Turkic',
							'wordUnigrams_similarity_Turkic',
							'POSBigrams_similarity_Turkic',
							'functionWords_similarity_Turkic',

							'charTrigrams_similarity_Indo-Aryan',
							'wordBigrams_similarity_Indo-Aryan',
							'wordUnigrams_similarity_Indo-Aryan',
							'POSBigrams_similarity_Indo-Aryan',
							'functionWords_similarity_Indo-Aryan',

							'charTrigrams_similarity_Native',
							'wordBigrams_similarity_Native',
							'wordUnigrams_similarity_Native',
							'POSBigrams_similarity_Native',
							'functionWords_similarity_Native',

							'charTrigrams_similarity_NonNative',
							'wordBigrams_similarity_NonNative',
							'wordUnigrams_similarity_NonNative',
							'POSBigrams_similarity_NonNative',
							'functionWords_similarity_NonNative',

							'charTrigrams_similarity_Reddit',
							'wordBigrams_similarity_Reddit',
							'wordUnigrams_similarity_Reddit',
							'POSBigrams_similarity_Reddit',
							'functionWords_similarity_Reddit',

							'charTrigrams_similarity_Twitter',
							'wordBigrams_similarity_Twitter',
							'wordUnigrams_similarity_Twitter',
							'POSBigrams_similarity_Twitter',
							'functionWords_similarity_Twitter'
							]

	origin = [
				'Native',
				'NonNative'
		]

	category = [
				'Reddit',
				'Twitter'
		]

	if(other == 'reddit'):
		lang = [
			'Bulgarian',
			'Croatian',
			'Czech',
			'Dutch',
			'English',
			'Finnish',
			'French', 
			'German', 
			'Italian',
			'Lithuanian',
			'Norwegian',
			'Polish',
			'Portuguese',
			'Romanian',
			'Russian', 
			'Serbian',
			'Slovene',
			'Spanish',
			'Swedish'
			]
		family = [
				'Balto-Slavic',
				'Germanic',
				'Romance',
				'Native'
		]
		
	elif(other == 'twitter'):
		lang = [
			'English',
			'French', 
			'German', 
			'Greek', 
			'Indian',
			'Japanese',
			'Russian', 
			'Turkish'
			]
		family = [
				'Balto-Slavic',
				'Germanic',
				'Greek',
				'Indo-Aryan',
				'Japonic',
				'Romance',
				'Turkic',
				'Native'
		]
	elif(other == 'combined'):
		lang = [
			'Bulgarian',
			'Croatian',
			'Czech',
			'Dutch',
			'English',
			'Finnish',
			'French', 
			'German', 
			'Greek', 
			'Indian',
			'Italian',
			'Japanese',
			'Lithuanian',
			'Norwegian',
			'Polish',
			'Portuguese',
			'Romanian',
			'Russian', 
			'Serbian',
			'Slovene',
			'Spanish',
			'Swedish',
			'Turkish'
			]
		family = [
				'Balto-Slavic',
				'Germanic',
				'Greek',
				'Indo-Aryan',
				'Japonic',
				'Romance',
				'Turkic',
				'Native'
		]
		
	

	if(source == 'reddit'):
		
		features = [	'elongated',
						'caps',
						'textLength',
						'sentenceWordLength',
						'spellDelta',
						  '#',
						  '@',
						  'E',
						',',
						  '~',
						  'U',
						'A',
						'D',
						'!',
						'N',
						'P',
						'O',
						'R',
						'&',
						  'L',
						  'Z',
						'^',
						'V',
						'$',
						'G',
						'T',
						  'X',
						  'S',
						  'Y',
						  'M',
						 'charTrigrams_similarity_French',
							'wordBigrams_similarity_French',
							'wordUnigrams_similarity_French',
							'POSBigrams_similarity_French',
							'functionWords_similarity_French',

							'charTrigrams_similarity_German',
							'wordBigrams_similarity_German',
							'wordUnigrams_similarity_German',
							'POSBigrams_similarity_German',
							'functionWords_similarity_German',

							'charTrigrams_similarity_Russian',
							'wordBigrams_similarity_Russian',
							'wordUnigrams_similarity_Russian',
							'POSBigrams_similarity_Russian',
							'functionWords_similarity_Russian',

							'charTrigrams_similarity_Bulgarian',
							'wordBigrams_similarity_Bulgarian',
							'wordUnigrams_similarity_Bulgarian',
							'POSBigrams_similarity_Bulgarian',
							'functionWords_similarity_Bulgarian',

							'charTrigrams_similarity_Croatian',
							'wordBigrams_similarity_Croatian',
							'wordUnigrams_similarity_Croatian',
							'POSBigrams_similarity_Croatian',
							'functionWords_similarity_Croatian',

							'charTrigrams_similarity_Czech',
							'wordBigrams_similarity_Czech',
							'wordUnigrams_similarity_Czech',
							'POSBigrams_similarity_Czech',
							'functionWords_similarity_Czech',

							'charTrigrams_similarity_Lithuanian',
							'wordBigrams_similarity_Lithuanian',
							'wordUnigrams_similarity_Lithuanian',
							'POSBigrams_similarity_Lithuanian',
							'functionWords_similarity_Lithuanian',

							'charTrigrams_similarity_Polish',
							'wordBigrams_similarity_Polish',
							'wordUnigrams_similarity_Polish',
							'POSBigrams_similarity_Polish',
							'functionWords_similarity_Polish',

							'charTrigrams_similarity_Serbian',
							'wordBigrams_similarity_Serbian',
							'wordUnigrams_similarity_Serbian',
							'POSBigrams_similarity_Serbian',
							'functionWords_similarity_Serbian',

							'charTrigrams_similarity_Slovene',
							'wordBigrams_similarity_Slovene',
							'wordUnigrams_similarity_Slovene',
							'POSBigrams_similarity_Slovene',
							'functionWords_similarity_Slovene',

							'charTrigrams_similarity_Finnish',
							'wordBigrams_similarity_Finnish',
							'wordUnigrams_similarity_Finnish',
							'POSBigrams_similarity_Finnish',
							'functionWords_similarity_Finnish',

							'charTrigrams_similarity_Dutch',
							'wordBigrams_similarity_Dutch',
							'wordUnigrams_similarity_Dutch',
							'POSBigrams_similarity_Dutch',
							'functionWords_similarity_Dutch',

							'charTrigrams_similarity_Norwegian',
							'wordBigrams_similarity_Norwegian',
							'wordUnigrams_similarity_Norwegian',
							'POSBigrams_similarity_Norwegian',
							'functionWords_similarity_Norwegian',

							'charTrigrams_similarity_Swedish',
							'wordBigrams_similarity_Swedish',
							'wordUnigrams_similarity_Swedish',
							'POSBigrams_similarity_Swedish',
							'functionWords_similarity_Swedish',

							'charTrigrams_similarity_Italian',
							'wordBigrams_similarity_Italian',
							'wordUnigrams_similarity_Italian',
							'POSBigrams_similarity_Italian',
							'functionWords_similarity_Italian',

							'charTrigrams_similarity_Spanish',
							'wordBigrams_similarity_Spanish',
							'wordUnigrams_similarity_Spanish',
							'POSBigrams_similarity_Spanish',
							'functionWords_similarity_Spanish',

							'charTrigrams_similarity_Portuguese',
							'wordBigrams_similarity_Portuguese',
							'wordUnigrams_similarity_Portuguese',
							'POSBigrams_similarity_Portuguese',
							'functionWords_similarity_Portuguese',

							'charTrigrams_similarity_Romanian',
							'wordBigrams_similarity_Romanian',
							'wordUnigrams_similarity_Romanian',
							'POSBigrams_similarity_Romanian',
							'functionWords_similarity_Romanian',

							'charTrigrams_similarity_Balto-Slavic',
							'wordBigrams_similarity_Balto-Slavic',
							'wordUnigrams_similarity_Balto-Slavic',
							'POSBigrams_similarity_Balto-Slavic',
							'functionWords_similarity_Balto-Slavic',

							'charTrigrams_similarity_Germanic',
							'wordBigrams_similarity_Germanic',
							'wordUnigrams_similarity_Germanic',
							'POSBigrams_similarity_Germanic',
							'functionWords_similarity_Germanic',

							'charTrigrams_similarity_Romance',
							'wordBigrams_similarity_Romance',
							'wordUnigrams_similarity_Romance',
							'POSBigrams_similarity_Romance',
							'functionWords_similarity_Romance',


							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English',
							'POSBigrams_similarity_English',
							'functionWords_similarity_English',

							'charTrigrams_similarity_Native',
							'wordBigrams_similarity_Native',
							'wordUnigrams_similarity_Native',
							'POSBigrams_similarity_Native',
							'functionWords_similarity_Native',

							'charTrigrams_similarity_NonNative',
							'wordBigrams_similarity_NonNative',
							'wordUnigrams_similarity_NonNative',
							'POSBigrams_similarity_NonNative',
							'functionWords_similarity_NonNative',

							'charTrigrams_similarity_Reddit',
							'wordBigrams_similarity_Reddit',
							'wordUnigrams_similarity_Reddit',
							'POSBigrams_similarity_Reddit',
							'functionWords_similarity_Reddit']
	elif(source == 'twitter'):
		
		features = [	'elongated',
						'caps',
						'textLength',
						'sentenceWordLength',
						'spellDelta',
						 '#',
						 '@',
						 'E',
						',',
						 '~',
						 'U',
						'A',
						'D',
						'!',
						'N',
						'P',
						'O',
						'R',
						'&',
						 'L',
						 'Z',
						'^',
						'V',
						'$',
						'G',
						'T',
						 'X',
						 'S',
						 'Y',
						 'M',
						'charTrigrams_similarity_French',
							'wordBigrams_similarity_French',
							'wordUnigrams_similarity_French',
							'POSBigrams_similarity_French',
							'functionWords_similarity_French',

							'charTrigrams_similarity_German',
							'wordBigrams_similarity_German',
							'wordUnigrams_similarity_German',
							'POSBigrams_similarity_German',
							'functionWords_similarity_German',

							'charTrigrams_similarity_Greek',
							'wordBigrams_similarity_Greek',
							'wordUnigrams_similarity_Greek',
							'POSBigrams_similarity_Greek',
							'functionWords_similarity_Greek',

							'charTrigrams_similarity_Indian',
							'wordBigrams_similarity_Indian',
							'wordUnigrams_similarity_Indian',
							'POSBigrams_similarity_Indian',
							'functionWords_similarity_Indian',

							'charTrigrams_similarity_Russian',
							'wordBigrams_similarity_Russian',
							'wordUnigrams_similarity_Russian',
							'POSBigrams_similarity_Russian',
							'functionWords_similarity_Russian',

							'charTrigrams_similarity_Japanese',
							'wordBigrams_similarity_Japanese',
							'wordUnigrams_similarity_Japanese',
							'POSBigrams_similarity_Japanese',
							'functionWords_similarity_Japanese',

							'charTrigrams_similarity_Turkish',
							'wordBigrams_similarity_Turkish',
							'wordUnigrams_similarity_Turkish',
							'POSBigrams_similarity_Turkish',
							'functionWords_similarity_Turkish',

							'charTrigrams_similarity_Balto-Slavic',
							'wordBigrams_similarity_Balto-Slavic',
							'wordUnigrams_similarity_Balto-Slavic',
							'POSBigrams_similarity_Balto-Slavic',
							'functionWords_similarity_Balto-Slavic',

							'charTrigrams_similarity_Germanic',
							'wordBigrams_similarity_Germanic',
							'wordUnigrams_similarity_Germanic',
							'POSBigrams_similarity_Germanic',
							'functionWords_similarity_Germanic',

							'charTrigrams_similarity_Romance',
							'wordBigrams_similarity_Romance',
							'wordUnigrams_similarity_Romance',
							'POSBigrams_similarity_Romance',
							'functionWords_similarity_Romance',

							'charTrigrams_similarity_Japonic',
							'wordBigrams_similarity_Japonic',
							'wordUnigrams_similarity_Japonic',
							'POSBigrams_similarity_Japonic',
							'functionWords_similarity_Japonic',

							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English',
							'POSBigrams_similarity_English',
							'functionWords_similarity_English',

							'charTrigrams_similarity_Turkic',
							'wordBigrams_similarity_Turkic',
							'wordUnigrams_similarity_Turkic',
							'POSBigrams_similarity_Turkic',
							'functionWords_similarity_Turkic',

							'charTrigrams_similarity_Indo-Aryan',
							'wordBigrams_similarity_Indo-Aryan',
							'wordUnigrams_similarity_Indo-Aryan',
							'POSBigrams_similarity_Indo-Aryan',
							'functionWords_similarity_Indo-Aryan',

							'charTrigrams_similarity_Native',
							'wordBigrams_similarity_Native',
							'wordUnigrams_similarity_Native',
							'POSBigrams_similarity_Native',
							'functionWords_similarity_Native',

							'charTrigrams_similarity_NonNative',
							'wordBigrams_similarity_NonNative',
							'wordUnigrams_similarity_NonNative',
							'POSBigrams_similarity_NonNative',
							'functionWords_similarity_NonNative',

							'charTrigrams_similarity_Twitter',
							'wordBigrams_similarity_Twitter',
							'wordUnigrams_similarity_Twitter',
							'POSBigrams_similarity_Twitter',
							'functionWords_similarity_Twitter']
	elif(source == 'combined'):
		
		features = [	'elongated',
						'caps',
						'textLength',
						'sentenceWordLength',
						'spellDelta',
						  '#',
						  '@',
						  'E',
						',',
						  '~',
						  'U',
						'A',
						'D',
						'!',
						'N',
						'P',
						'O',
						'R',
						'&',
						  'L',
						  'Z',
						'^',
						'V',
						'$',
						'G',
						'T',
						  'X',
						 'S',
						  'Y',
						  'M',
						  'charTrigrams_similarity_French',
							'wordBigrams_similarity_French',
							'wordUnigrams_similarity_French',
							'POSBigrams_similarity_French',
							'functionWords_similarity_French',

							'charTrigrams_similarity_German',
							'wordBigrams_similarity_German',
							'wordUnigrams_similarity_German',
							'POSBigrams_similarity_German',
							'functionWords_similarity_German',

							'charTrigrams_similarity_Greek',
							'wordBigrams_similarity_Greek',
							'wordUnigrams_similarity_Greek',
							'POSBigrams_similarity_Greek',
							'functionWords_similarity_Greek',

							'charTrigrams_similarity_Indian',
							'wordBigrams_similarity_Indian',
							'wordUnigrams_similarity_Indian',
							'POSBigrams_similarity_Indian',
							'functionWords_similarity_Indian',

							'charTrigrams_similarity_Russian',
							'wordBigrams_similarity_Russian',
							'wordUnigrams_similarity_Russian',
							'POSBigrams_similarity_Russian',
							'functionWords_similarity_Russian',

							'charTrigrams_similarity_Japanese',
							'wordBigrams_similarity_Japanese',
							'wordUnigrams_similarity_Japanese',
							'POSBigrams_similarity_Japanese',
							'functionWords_similarity_Japanese',

							'charTrigrams_similarity_Turkish',
							'wordBigrams_similarity_Turkish',
							'wordUnigrams_similarity_Turkish',
							'POSBigrams_similarity_Turkish',
							'functionWords_similarity_Turkish',

							'charTrigrams_similarity_Bulgarian',
							'wordBigrams_similarity_Bulgarian',
							'wordUnigrams_similarity_Bulgarian',
							'POSBigrams_similarity_Bulgarian',
							'functionWords_similarity_Bulgarian',

							'charTrigrams_similarity_Croatian',
							'wordBigrams_similarity_Croatian',
							'wordUnigrams_similarity_Croatian',
							'POSBigrams_similarity_Croatian',
							'functionWords_similarity_Croatian',

							'charTrigrams_similarity_Czech',
							'wordBigrams_similarity_Czech',
							'wordUnigrams_similarity_Czech',
							'POSBigrams_similarity_Czech',
							'functionWords_similarity_Czech',

							'charTrigrams_similarity_Lithuanian',
							'wordBigrams_similarity_Lithuanian',
							'wordUnigrams_similarity_Lithuanian',
							'POSBigrams_similarity_Lithuanian',
							'functionWords_similarity_Lithuanian',

							'charTrigrams_similarity_Polish',
							'wordBigrams_similarity_Polish',
							'wordUnigrams_similarity_Polish',
							'POSBigrams_similarity_Polish',
							'functionWords_similarity_Polish',

							'charTrigrams_similarity_Serbian',
							'wordBigrams_similarity_Serbian',
							'wordUnigrams_similarity_Serbian',
							'POSBigrams_similarity_Serbian',
							'functionWords_similarity_Serbian',

							'charTrigrams_similarity_Slovene',
							'wordBigrams_similarity_Slovene',
							'wordUnigrams_similarity_Slovene',
							'POSBigrams_similarity_Slovene',
							'functionWords_similarity_Slovene',

							'charTrigrams_similarity_Finnish',
							'wordBigrams_similarity_Finnish',
							'wordUnigrams_similarity_Finnish',
							'POSBigrams_similarity_Finnish',
							'functionWords_similarity_Finnish',

							'charTrigrams_similarity_Dutch',
							'wordBigrams_similarity_Dutch',
							'wordUnigrams_similarity_Dutch',
							'POSBigrams_similarity_Dutch',
							'functionWords_similarity_Dutch',

							'charTrigrams_similarity_Norwegian',
							'wordBigrams_similarity_Norwegian',
							'wordUnigrams_similarity_Norwegian',
							'POSBigrams_similarity_Norwegian',
							'functionWords_similarity_Norwegian',

							'charTrigrams_similarity_Swedish',
							'wordBigrams_similarity_Swedish',
							'wordUnigrams_similarity_Swedish',
							'POSBigrams_similarity_Swedish',
							'functionWords_similarity_Swedish',

							'charTrigrams_similarity_Italian',
							'wordBigrams_similarity_Italian',
							'wordUnigrams_similarity_Italian',
							'POSBigrams_similarity_Italian',
							'functionWords_similarity_Italian',

							'charTrigrams_similarity_Spanish',
							'wordBigrams_similarity_Spanish',
							'wordUnigrams_similarity_Spanish',
							'POSBigrams_similarity_Spanish',
							'functionWords_similarity_Spanish',

							'charTrigrams_similarity_Portuguese',
							'wordBigrams_similarity_Portuguese',
							'wordUnigrams_similarity_Portuguese',
							'POSBigrams_similarity_Portuguese',
							'functionWords_similarity_Portuguese',

							'charTrigrams_similarity_Romanian',
							'wordBigrams_similarity_Romanian',
							'wordUnigrams_similarity_Romanian',
							'POSBigrams_similarity_Romanian',
							'functionWords_similarity_Romanian',

							'charTrigrams_similarity_Balto-Slavic',
							'wordBigrams_similarity_Balto-Slavic',
							'wordUnigrams_similarity_Balto-Slavic',
							'POSBigrams_similarity_Balto-Slavic',
							'functionWords_similarity_Balto-Slavic',

							'charTrigrams_similarity_Germanic',
							'wordBigrams_similarity_Germanic',
							'wordUnigrams_similarity_Germanic',
							'POSBigrams_similarity_Germanic',
							'functionWords_similarity_Germanic',

							'charTrigrams_similarity_Romance',
							'wordBigrams_similarity_Romance',
							'wordUnigrams_similarity_Romance',
							'POSBigrams_similarity_Romance',
							'functionWords_similarity_Romance',

							'charTrigrams_similarity_Japonic',
							'wordBigrams_similarity_Japonic',
							'wordUnigrams_similarity_Japonic',
							'POSBigrams_similarity_Japonic',
							'functionWords_similarity_Japonic',

							'charTrigrams_similarity_English',
							'wordBigrams_similarity_English',
							'wordUnigrams_similarity_English',
							'POSBigrams_similarity_English',
							'functionWords_similarity_English',

							'charTrigrams_similarity_Turkic',
							'wordBigrams_similarity_Turkic',
							'wordUnigrams_similarity_Turkic',
							'POSBigrams_similarity_Turkic',
							'functionWords_similarity_Turkic',

							'charTrigrams_similarity_Indo-Aryan',
							'wordBigrams_similarity_Indo-Aryan',
							'wordUnigrams_similarity_Indo-Aryan',
							'POSBigrams_similarity_Indo-Aryan',
							'functionWords_similarity_Indo-Aryan',

							'charTrigrams_similarity_Native',
							'wordBigrams_similarity_Native',
							'wordUnigrams_similarity_Native',
							'POSBigrams_similarity_Native',
							'functionWords_similarity_Native',

							'charTrigrams_similarity_NonNative',
							'wordBigrams_similarity_NonNative',
							'wordUnigrams_similarity_NonNative',
							'POSBigrams_similarity_NonNative',
							'functionWords_similarity_NonNative',

							'charTrigrams_similarity_Reddit',
							'wordBigrams_similarity_Reddit',
							'wordUnigrams_similarity_Reddit',
							'POSBigrams_similarity_Reddit',
							'functionWords_similarity_Reddit',

							'charTrigrams_similarity_Twitter',
							'wordBigrams_similarity_Twitter',
							'wordUnigrams_similarity_Twitter',
							'POSBigrams_similarity_Twitter',
							'functionWords_similarity_Twitter']

	filename = 'classification/feature_average_'+source+'.csv'

	data.groupby(['lang']).mean().to_csv(filename)
	data.groupby(['langFam']).mean().to_csv(filename, mode='a')
	data.groupby(['category']).mean().to_csv(filename, mode='a')
	data.groupby(['origin']).mean().to_csv(filename, mode='a')

	filename = 'classification/feature_max_'+source+'.csv'

	data.groupby(['lang'])[features].max().to_csv(filename)
	data.groupby(['langFam'])[features].max().to_csv(filename, mode='a')
	data.groupby(['category'])[features].max().to_csv(filename, mode='a')
	data.groupby(['origin'])[features].max().to_csv(filename, mode='a')

	filename = 'classification/feature_std_'+source+'.csv'

	data.groupby(['lang']).std().to_csv(filename)
	data.groupby(['langFam']).std().to_csv(filename, mode='a')
	data.groupby(['category']).std().to_csv(filename, mode='a')
	data.groupby(['origin']).std().to_csv(filename, mode='a')


	#filename = 'classification/feature_min_'+source+'.csv'

	#data.groupby(['lang'])[features].min().to_csv(filename)
	#data.groupby(['langFam'])[features].min().to_csv(filename, mode='a')
	#data.groupby(['category'])[features].min().to_csv(filename, mode='a')
	#data.groupby(['origin'])[features].min().to_csv(filename, mode='a')




if __name__ == "__main__":
	file = sys.argv[1]
	filters = 'none'
	arg2 = 'none'
	arg3 = 'none'
	if(len(sys.argv) > 2):
		func = sys.argv[2]
	if(len(sys.argv) > 3):
		filters = sys.argv[3]
	if(len(sys.argv) > 4):
		arg2 = sys.argv[4]
	if(len(sys.argv) > 5):
		arg3 = sys.argv[5]
	if(func == 'tag'):
		print('Tagging '+ file + ' with filters: '+filters+', '+arg2)
		tagCSV(file, filters, arg2, arg3)
	elif(func == 'split'):
		print('Splitting '+ file + ' with size: ' +filters + ' for '+arg2)
		splitFile(file,filters, arg2)
	elif(func == 'ngram'):
		print('Creating ngram model '+ file + ' for '+filters)
		ngramModel(file,filters)
	elif(func == 'classifier'):
		print('Creating classification file for ' +filters)
		createClassifierFile(file,filters)
	elif(func == 'reformat'):
		print('Reformatting for '+filters)
		reformatCSV(file,filters, arg2)
	elif(func == 'average'):
		print('Averaging functions for '+filters)
		calculateFeatureAverage(file, filters, arg2)
	elif(func == 'word2vec'):
		print('Word2Vec for '+filters)
		word2VecModel(file, filters)
	print('Done')
	#print(result)