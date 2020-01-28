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
#	OUTPUT:		list of features
#					+elongated word count
#					+text length (not counting punctuation)
#					+mean word length
#					+spelling delta (difference between corrected and original words e.g. helo vs. hello)
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


	print(sentence)
	sentenceCharNGrams = Counter(ngrams(sentence, 3))
	print(sentenceCharNGrams)

	return {'elongated' : countElongated, 'sentenceLength': sentenceLength, 'sentenceWordLength' : sentenceWordLength, 'spellDelta':sentenceSpellDelta}


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
#	=================================================================	

def analyzeText(file, family='none', lang='none'):
	#txt_Raw = []
	filteredText = []
	txt_Occurence = {}

	txt_Import = pd.read_csv(file, header=0,  nrows=10, sep=',', usecols=['user','post'])

	for text in txt_Import['post'].values:
		#txt_Raw.append(formatText(text))
		text = formatText(text)
		if(isLanguage(text, 'English')):
			filteredText.append(text)

	for text in filteredText:
		extractFeatures(text)

	


	text_POS = runPOSTagger(filteredText)
	

	num_tweet = 0
	for i in txt_POSTagged:
		if(num_tweet < len(txt_English)):
			txt_Occurence[num_tweet] = {'len':0, 'wordlen':0, '#':0, '@':0, 'E':0, ',':0, '~':0, 'U':0, 'A':0, 'D':0, '!':0, 'N':0, 'P':0, 'O':0, 'R':0, '&':0, 'L':0, 'Z':0, '^':0, 'V':0, '$':0, 'G':0, 'T':0, 'X':0, 'S':0, 'Y':0, 'M':0 ,'elong':0, 'langFam': family, 'lang': lang, 'user':" ", 'spellDelta':0, 'ngrams':[]}
			txt_Occurence[num_tweet]['langFam'] = family
			txt_Occurence[num_tweet]['user'] = txt_Import.at[num_tweet, 'user']

			spellDeltaASPELL = []
			spellDeltaLANGCHECK = []
			
			sum_wordlen = 0 
			for j in i:
				txt_Occurence[num_tweet]['wordlen'] = txt_Occurence[num_tweet]['wordlen'] + len(j[0])
				sum_wordlen = sum_wordlen + len(j[0])
				key = j[1]
				if key in txt_Occurence[num_tweet]:
					txt_Occurence[num_tweet][key] = txt_Occurence[num_tweet][key] + 1
					#if 'len' in txt_Occurence[num_tweet]:
					#	if key is not ",":
					#		txt_Occurence[num_tweet]['len'] = txt_Occurence[num_tweet]['len'] + 1
					#else:
					#	txt_Occurence[num_tweet]['len'] = 1
				else:
					txt_Occurence[num_tweet][key] = 1


			characterNGram = txt_English[num_tweet].lower().split()
			#characterNGram = re.sub(r'[^a-zA-Z0-9\s]', '', characterNGram)
			char3Gram_count = Counter(ngrams(characterNGram, 2))
			txt_Occurence[num_tweet]['ngrams'] = char3Gram_count.most_common(5)


			num_tweet = num_tweet + 1

			

	for x in txt_Occurence:
		#print(txt_Occurence[x])
		for z in txt_Occurence[x]:
			if not(z == 'lang' or z == 'langFam' or z == 'len' or z == 'user' or z == 'spellDelta' or z == 'ngrams' or z == 'wordlen'): 
				txt_Occurence[x][z] = (txt_Occurence[x][z] / max(1,txt_Occurence[x]['len'])) * 100
			elif(z == 'wordlen'):
				txt_Occurence[x][z] = (txt_Occurence[x][z] / sum_wordlen)
			#print(x,z,txt_Occurence[x][z])

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
	fields = [ 'len', 'wordlen', '#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'elong', 'langFam', 'lang', 'user', 'spellDelta', 'ngrams']
	#print(txt_Occurence)
	with open('result'+txt+ '.csv', "w") as f:
	    w = csv.DictWriter(f, fields)
	    w.writeheader()
	    for k in txt_Occurence:
	        #print(txt_Occurence[k])
	        #w.writerow({field: txt_Occurence[k].get(field) or k for field in fields})
	        w.writerow(txt_Occurence[k])
	return txt_Occurence


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
	#print(result)