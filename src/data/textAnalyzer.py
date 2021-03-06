import sys
import os
import csv
import pandas as pd
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '../tools'))
import CMUTweetTagger
from polyglot.detect import Detector
from polyglot.text import Text
import aspell
import Levenshtein as ls
import language_check
import techniques
import re
import glob
import stanza
import nltk
from tokenizer import tokenizer


from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 



#	=================================================================
#							formatText
#
#	INPUT:		any text
#
#	OUTPUT:		formatted text with unicode, spaces between letters/punctuation and abbreviations removed
#	=================================================================	

def formatText(text):
	text = str(text)
	text = text.replace("\\n", "  ")
	text = text.replace("\\'t", "'t")
	text = text.replace("\\ 's", "'s")
	text = text.replace("\\ 'r", "'r")
	text = text.replace("\\ 'l", "'l")
	text = text.replace("\\'l", "'l")
	text = text.replace("\\ 'v", "'v")
	text = text.replace("\ ", "")
	text = text.replace('\\ " ', '"')
	text = text.replace("''", ' "')
	text = text.replace("``", '"')
	text = text.replace(" n'", "n'")
	text = text.replace(" 'r", "'r")
	text = text.replace("gon na", "gonna")
	text = text.replace(" 'll", "'ll")
	text = text.replace(": )", ":)")
	text = text.replace(" : - )", ":-)")
	text = text.replace("s : ", "s:")
	text = text.replace("p : ", "p:")
	text = text.replace(" : ", ": ")
	text = text.replace(" :/", ":/")
	text = text.replace("? !", "?!")
	text = text.replace("! ?", "!?")
	text = text.replace("I '", "I'" )
	text = text.replace(" 's", "'s")
	text = text.replace ("& lt ;", "<")
	text = text.replace("& gt ;", ">")
	text = text.replace("&gt", ">")
	text = text.replace("& amp", "&")
	text = text.replace(" # ", "#")
	text = text.replace(" % ", "% ")
	text = text.replace("**", "")
	text = text.replace("\\'", "'")
	text = text.replace("^^^", "")
	text = text.replace("n\\", "n")
	text = text.replace("p . s .", "p.s.")
	text = techniques.removeUnicode(text)
	#print(text)
	return text

#	=================================================================
#							isLanguage
#
#	INPUT:		any string
#					opt-- language
#
#	OUTPUT:		boolean check for given language (default is 'English')
#	=================================================================	


def isLanguage(text, language='English'):
	return Text(text).language.name == language

#	=================================================================
#							runPOSTagger
#
#	INPUT:		any string or list of strings
#
#	OUTPUT:		list of POS tags for given string(s)
#	=================================================================	

def runPOSTagger(text):
	return CMUTweetTagger.runtagger_parse(text)

#	=================================================================
#							extractFeatures
#
#	INPUT:		any string
#					opt-- language array (abbreviation and specification e.g. 'en' and 'en-US')
#
#	OUTPUT:		dict of features
#					+corrected sentence
#					+elongated word count
#					+text length (not counting punctuation)
#					+mean word length
#					+spelling delta (difference between corrected and original words e.g. helo vs. hello)
#					+char 3grams
#					+word 2grams
#	=================================================================	

def extractFeatures(text, lang=['en','en-US']):
	
	hashtags = techniques.extractHashtags(text)
	urls = techniques.extractURL(text)
	atUsers = techniques.extractAtUser(text)
	text = techniques.removeHashtags(text)
	text = techniques.removeAtUser(text)
	text = techniques.replaceURL(text, '')

	porter = PorterStemmer()

	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(text) 
	filtered_words = [] 
	functionWords = []
	  
	for w in word_tokens: 
	    if w not in stop_words:
	        filtered_words.append(w)
	    else:
	    	functionWords.append(w)
	
	#print(filtered_words)
	
	countElongated = techniques.countElongated(text)
	#sentenceWords = techniques.words(text)
	filtered_Sentence = ' '.join(filtered_words)
	originalText = text

	countCaps = techniques.countAllCaps(text)
	

	s = aspell.Speller('lang', lang[0])
	lang_tool = language_check.LanguageTool(lang[1])

	aspellDelta = []
	for word in filtered_words:
		suggest = s.suggest(word)
		if(len(suggest) > 0):
			suggest = suggest[0]
		else:
			suggest = word
		aspellDelta.append(ls.distance(word, suggest))


	
	
	langCheckDelta = []
	correctedSentence = techniques.words(language_check.correct(filtered_Sentence, lang_tool.check(filtered_Sentence)))
	word = 0
	while word < min(len(correctedSentence), len(filtered_words)):
		langCheckDelta.append(ls.distance(correctedSentence[word], filtered_words[word]))
		word += 1

	aspellDelta = sum(aspellDelta) / max(1,len(aspellDelta))
	langCheckDelta = sum(langCheckDelta) / max(1,len(langCheckDelta))
	sentenceSpellDelta = (aspellDelta + langCheckDelta) / 2.0

	sentence = ' '.join(correctedSentence)
	sentenceLength = len(filtered_words)
	sentenceWordLength = sum([len(x) for x in filtered_words]) / max(1,sentenceLength)

	stemmed_words = []
	for word in filtered_words:
		if word not in stop_words: 
			stemmed_words.append(porter.stem(word))
		else:
			functionWords.append(word)
		
	stemmed_sentence = ' '.join(stemmed_words)

	tokenizer = RegexpTokenizer("[a-zA-Z]+")
	tokenizedText = tokenizer.tokenize(stemmed_sentence)
	tokenizedChar = [c for c in ' '.join(tokenizedText)]

	sentenceCharTrigrams = [ ''.join(grams) for grams in ngrams(tokenizedChar, 3)]
	sentenceWordBigrams = [ ' '.join(grams) for grams in ngrams(tokenizedText, 2)]
	sentenceWordUnigrams = [ ' '.join(grams) for grams in ngrams(tokenizedText, 1)]
	functionWords = [ ' '.join(grams) for grams in ngrams(functionWords, 1)]

	return {'correctedSentence': sentence, 'originalSentence': originalText,'filteredSentence':filtered_Sentence, 'stemmedSentence':stemmed_sentence, 'elongated' : countElongated, 'caps': countCaps, 'textLength': sentenceLength, 'sentenceWordLength' : sentenceWordLength, 'spellDelta':sentenceSpellDelta, 'charTrigrams':sentenceCharTrigrams, 'wordBigrams': sentenceWordBigrams, 'wordUnigrams':sentenceWordUnigrams, 'POSBigrams': '', 'functionWords': functionWords, 'hashtag': hashtags, 'url':urls}


#	=================================================================
#							analyzeText
#
#	INPUT:		csv file with columns 'user' and 'post' and separated by ','
#					opt-- language family for tagging (e.g. 'Balto-Slavic')
#					opt-- native language for tagging (e.g. 'Polish')
#					opt-- limit number of imported rows
#
#	OUTPUT:		csv file with text features
#						+POS-tags
#						+text length (not counting punctuation)
#						+mean word length
#						+elongation rate
#						+language family
#						+language
#						+username
#						+spelling delta (difference between corrected and original words e.g. helo vs. hello)
#						+character 3-grams
#						+word 2-grams
#	=================================================================	

def analyzeText(file, filetype):
	textFiltered = []
	textImported = pd.read_csv(file, header=None, sep=',', skiprows=0, encoding="utf-8-sig")
	textImported.columns = ['text', 'lang', 'langFam', 'category', 'origin']
	textImported = textImported[textImported.langFam.str.contains('langFam') == False].reset_index()

	num_row = 0	

	nltk.download('stopwords')
	nltk.download('punkt')
	
	while num_row < len(textImported):
		textPost = textImported.at[num_row, 'text']
		lang = textImported.at[num_row, 'lang']
		family = textImported.at[num_row, 'langFam']
		category = filetype
		origin = textImported.at[num_row, 'origin']


		text = formatText(textPost)

		textFiltered.append([extractFeatures(text), {'langFam': family, 'lang': lang, 'category':category, 'origin':origin}])
		num_row += 1
		if((num_row % 100) == 0):
			print(str(num_row)+' / '+str(len(textImported)))

	print(textFiltered)
	text_POS = []
	num_row = 0
	if(filetype == 'reddit'):
		stanza.download('en')
		nlp = stanza.Pipeline('en', processors='tokenize,pos')
		R = tokenizer.RedditTokenizer()
		for text in textFiltered:
			tokens = R.tokenize(text[0]['stemmedSentence'])
			current = []
			for word in tokens:
				doc = nlp(word)
				for sentence in doc.sentences:
					for word in sentence.words:
						postag = 'none'
						if(word.pos == 'ADV'):
							postag = 'R'
						elif(word.pos == 'NOUN'):
							postag = 'N'
						elif(word.pos == 'CCONJ'):
							postag = '&'
						elif(word.pos == 'DET'):
							postag = 'D'
						elif(word.pos == 'INTJ'):
							postag = '!'
						elif(word.pos == 'NUM'):
							postag = '$'
						elif(word.pos == 'PART'):
							postag = 'T'
						elif(word.pos == 'PROPN'):
							postag = '^'
						elif(word.pos == 'PUNCT'):
							postag = ','
						elif(word.pos == 'ADJ'):
							postag = 'A'
						elif(word.pos == 'X' or word.pos == 'SYM'):
							postag = 'G'
						elif(word.pos == 'AUX'):
							postag = 'V'
						elif(word.pos == 'ADP'):
							postag = 'P'
						elif(word.pos == 'PRON'):
							postag = 'O'
						elif(word.pos == 'SCONJ'):
							postag = 'P'
						elif(word.pos == 'VERB'):
							postag = 'V'
						else:
							print(word.pos)
						current.append([word.text, postag])
			text_POS.append(current)
			num_row += 1
			if((num_row % 100) == 0):
				print(str(num_row)+' / '+str(len(textFiltered)))

	else:
		text_POS = runPOSTagger(text[0]['stemmedSentence'] for text in textFiltered)


	
	textFiltered = textFiltered[:len(text_POS)] 

	#	N : Common Noun
	#	O : Pronoun
	#	S : nominal + possessive
	#	^ : Proper Noun
	#	Z : proper noun + possessive
	#	L : nominal + verb
	#	M : proper noun + verbal 
	#	V : verb
	#	A : adjective
	#	R : adverb
	#	! : interjection
	#	T : verb particle
	#	X : existential there, predeterminers
	#	Y : X + verbal
	#	D : determiner
	#	P : pre- or postposition
	#	& : coordinating conjunction
	#	E : emoticon
	#	U : URL or email address
	#	~ : discourse marker (retweet)
	#	@ : at-mention
	#	# : hashtag
	#	$ : numeral
	#	, : punctuation
	#	G : other abbreviations, foreign words, possessive endings, symbols, garbage

	num_tweet = 0
	while num_tweet < len(textFiltered):
		textFiltered[num_tweet].append({'#':0, '@':0, 'E':0, ',':0, '~':0, 'U':0, 'A':0, 'D':0, '!':0, 'N':0, 'P':0, 'O':0, 'R':0, '&':0, 'L':0, 'Z':0, '^':0, 'V':0, '$':0, 'G':0, 'T':0, 'X':0, 'S':0, 'Y':0, 'M':0 })
		#print([x[1] for x in text_POS[num_tweet]])
		textFiltered[num_tweet][0]['POSBigrams'] = [ ' '.join(grams) for grams in ngrams([x[1] for x in text_POS[num_tweet]], 2)]
		for tag in text_POS[num_tweet]:
			key = tag[1]
			if key in textFiltered[num_tweet][2]:
				textFiltered[num_tweet][2][key] += 1 / max(1, textFiltered[num_tweet][0]['textLength'])
			else:
				textFiltered[num_tweet][2][key] = 1 / max(1, textFiltered[num_tweet][0]['textLength'])


		textFiltered[num_tweet][2]['#'] = len(textFiltered[num_tweet][0]['hashtag'])
		textFiltered[num_tweet][2]['U'] = len(textFiltered[num_tweet][0]['url'])
		num_tweet += 1

	

	return textFiltered


if __name__ == "__main__":
	path = sys.argv[1]
	filetype = sys.argv[2]
	#family = 'none'
	#lang = 'none'
	#limit = 60000
	if(len(sys.argv) > 3):
		family = sys.argv[3]
	if(len(sys.argv) > 4):
		lang = sys.argv[4]
	if(len(sys.argv) > 5):
		category = sys.argv[5]
	if(len(sys.argv) > 6):
		limit = sys.argv[6]

	outputValues = []
	#if(filetype == 'reddit'):
	#print('Running '+ path + ', '+family+', '+lang+', '+str(limit))
	outputValues.append(analyzeText(path, filetype))
	#else:
	#	files = glob.glob(path+'*.csv')
	#	for file in files:
	#		print('Running '+ file + ', '+family+', '+lang+', '+str(limit))
	#		outputValues.append(analyzeText(file, filetype, family, lang, category, limit))


	fields = [ 'correctedSentence','originalSentence','filteredSentence','stemmedSentence','elongated','caps','textLength','sentenceWordLength','spellDelta','charTrigrams','wordBigrams','wordUnigrams','POSBigrams','functionWords','hashtag','url','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'category', 'origin']
	filename = 'output/result_anaylze.csv'
	os.makedirs(os.path.dirname(filename), exist_ok=True)
	with open(filename, "w") as f:
		w = csv.DictWriter(f, fields)
		w.writeheader()
		for results in outputValues:
			for output in results:
				flatList = {**output[0], **output[1], **output[2]}
				w.writerow(flatList)
	print('Done')
