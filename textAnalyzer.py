import CMUTweetTagger
import sys
import os
import csv
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath('text-preprocessing-techniques'))
from polyglot.detect import Detector
from polyglot.text import Text
import aspell
import Levenshtein as ls
import language_check
import techniques
import re
from collections import Counter
from nltk.util import ngrams


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
	text = text.replace("\ ", "")
	text = text.replace("''", ' "')
	text = text.replace("``", '"')
	text = text.replace(" n'", "n'")
	text = text.replace(" 'r", "'r")
	text = text.replace("gon na", "gonna")
	text = text.replace(" 'll", "'ll")
	text = text.replace(" : )", ":)")
	text = text.replace(" : - )", ":-)")
	text = text.replace(" : ", ": ")
	text = text.replace(" :/", ":/")
	text = text.replace("? !", "?!")
	text = text.replace("! ?", "!?")
	text = text.replace("I '", "I'" )
	text = text.replace(" 's", "'s")
	text = text.replace ("& lt ;", "<")
	text = text.replace("& gt ;", ">")
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
	countElongated = techniques.countElongated(text)
	sentenceWords = techniques.words(text)
	sentence = ' '.join(sentenceWords)
	


	s = aspell.Speller('lang', lang[0])
	lang_tool = language_check.LanguageTool(lang[1])

	aspellDelta = []
	for word in sentenceWords:
		suggest = s.suggest(word)
		if(len(suggest) > 0):
			suggest = suggest[0]
		else:
			suggest = word
		aspellDelta.append(ls.distance(word, suggest))
	
	langCheckDelta = []
	correctedSentence = techniques.words(language_check.correct(sentence, lang_tool.check(sentence)))
	word = 0
	while word < min(len(correctedSentence), len(sentenceWords)):
		langCheckDelta.append(ls.distance(correctedSentence[word], sentenceWords[word]))
		word += 1

	aspellDelta = sum(aspellDelta) / len(aspellDelta)
	langCheckDelta = sum(langCheckDelta) / len(langCheckDelta)
	sentenceSpellDelta = (aspellDelta + langCheckDelta) / 2.0

	sentence = ' '.join(correctedSentence)
	sentenceLength = len(correctedSentence)
	sentenceWordLength = sum(map(len, [x for x in correctedSentence])) / sentenceLength


	sentenceCharNGrams = Counter(ngrams(sentence, 3)).most_common(5)
	sentenceWordNGrams = Counter(ngrams(sentenceWords, 2)).most_common(5)

	return {'correctedSentence': sentence,'elongated' : countElongated, 'sentenceLength': sentenceLength, 'sentenceWordLength' : sentenceWordLength, 'spellDelta':sentenceSpellDelta, 'charNGrams':sentenceCharNGrams, 'wordNGrams': sentenceWordNGrams}


#	=================================================================
#							analyzeText
#
#	INPUT:		csv file with columns 'user' and 'post' and separated by ','
#					opt-- language family for tagging (e.g. 'Balto-Slavic')
#					opt-- native language for tagging (e.g. 'Polish')
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

def analyzeText(file, family='none', lang='none'):
	textFiltered = []
	textImported = pd.read_csv(file, header=0,  nrows=10, sep=',', usecols=['user','post'])

	for text in textImported['post'].values:
		text = formatText(text)
		if(isLanguage(text, 'English')):
			textFiltered.append([extractFeatures(text)])

	text_POS = runPOSTagger(text[0]['correctedSentence'] for text in textFiltered)

	num_tweet = 0
	while num_tweet < len(textImported):
		textFiltered[num_tweet].append({'#':0, '@':0, 'E':0, ',':0, '~':0, 'U':0, 'A':0, 'D':0, '!':0, 'N':0, 'P':0, 'O':0, 'R':0, '&':0, 'L':0, 'Z':0, '^':0, 'V':0, '$':0, 'G':0, 'T':0, 'X':0, 'S':0, 'Y':0, 'M':0 })
		textFiltered[num_tweet].append({'langFam': family, 'lang': lang, 'user':textImported.at[num_tweet, 'user']})

		for tag in text_POS[num_tweet]:
			key = tag[1]
			if key in textFiltered[num_tweet][1]:
				textFiltered[num_tweet][1][key] += 1 / max(1, textFiltered[num_tweet][0]['sentenceLength']) * 100
			else:
				textFiltered[num_tweet][1][key] = 1 / max(1, textFiltered[num_tweet][0]['sentenceLength']) * 100

		num_tweet += 1

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

	fields = [ 'correctedSentence','elongated','sentenceLength','sentenceWordLength','spellDelta','charNGrams','wordNGrams','#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'langFam', 'lang', 'user']
	with open('result'+file.split('/')[-1].split('.')[1]+file.split('/')[-1].split('.')[3]+ '.csv', "w") as f:
		w = csv.DictWriter(f, fields)
		w.writeheader()
		for output in textFiltered:
			flatList = {**output[0], **output[1], **output[2]}
			w.writerow(flatList)


if __name__ == "__main__":
	file = sys.argv[1]
	family = 'none'
	lang = 'none'
	if(len(sys.argv) > 2):
		family = sys.argv[2]
	if(len(sys.argv) > 3):
		lang = sys.argv[3]
	
	print('Running '+ file + ', '+family+', '+lang)
	result = analyzeText(file, family, lang)
	print('Done')