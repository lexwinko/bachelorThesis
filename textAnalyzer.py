import CMUTweetTagger
import sys
import os
import csv
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath('../text-preprocessing-techniques'))
from polyglot.detect import Detector
from polyglot.text import Text
import aspell
import Levenshtein as ls
import language_check
import techniques
import re
from collections import Counter
from nltk.util import ngrams

hashtag = "None"

def analyzeText(txt, family, lang):
	txt_Raw = []
	txt_English = []
	txt_Occurence = {}
	#with open(txt, newline='') as csvFile:
	#	csvReader = csv.reader(csvFile, delimiter=';')
	#	for row in csvReader:
	#		txt_Raw.append((" ".join(row)).replace('\n', " "))
	txt_Import = pd.read_csv(txt+".csv", header=0,  sep=',', usecols=['user','post'])
	#print(txt_Import)
	for text in txt_Import['post'].values:
		#print(text)
		text = str(text)
		text = text.replace("\\n", "  ")
		text = text.replace("\ ", "")
		text = text.replace("''", ' "')
		text = text.replace("``", '"')
		#text = text.replace(' "', '"')
		text = text.replace(" n'", "n'")
		text = text.replace(" 'r", "'r")
		text = text.replace("gon na", "gonna")
		text = text.replace(" 'll", "'ll")
		text = text.replace(" : ", ":")
		text = text.replace(" :/", ":/")
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
		text = techniques.removeUnicode(text)

		#print(text)
		#decoded = text
		#try:
		#	decoded = text.encode('utf-8','surrogateescape').decode('unicode-escape', 'surrogateescape')
		#except:
		#	continue
		#print(decoded)
		txt_Raw.append(text)
	#counter = 0
	#with open("txtRaw_"+hashtag+".txt", "a") as result:
	#	for x in txt_Raw:
	#		result.write(str(counter)+"\t"+x+"\t"+" \n")
	#		counter = counter + 1
	#print(txt_Raw)

	for text in txt_Raw:
		if(Text(text).language.name == "English"):
			txt_English.append(text)


	s = aspell.Speller('lang', 'en')
	lang_tool = language_check.LanguageTool('en-US')


	txt_POSTagged = CMUTweetTagger.runtagger_parse(txt_English)
	
	#with open("resultAnalyzingPOS.csv", "w") as f:
	#	for x in txt_POSTagged:
	#		#print(x)
	#		for y in x:
	#			f.write(y[0]+";"+y[1]+";"+str(y[2])+"\n")
	num_tweet = 0
	for i in txt_POSTagged:
		if(num_tweet < len(txt_English)):
			#print(len(txt_English), num_tweet, len(txt_POSTagged))
			#print(str(num_tweet) + ' / ' + str(len(txt_POSTagged)))
			txt_Occurence[num_tweet] = {'len':0, 'wordlen':0, '#':0, '@':0, 'E':0, ',':0, '~':0, 'U':0, 'A':0, 'D':0, '!':0, 'N':0, 'P':0, 'O':0, 'R':0, '&':0, 'L':0, 'Z':0, '^':0, 'V':0, '$':0, 'G':0, 'T':0, 'X':0, 'S':0, 'Y':0, 'M':0 ,'elong':0, 'langFam': family, 'lang': lang, 'user':" ", 'spellDelta':0, 'ngrams':[]}
			txt_Occurence[num_tweet]['langFam'] = family
			txt_Occurence[num_tweet]['lang'] = lang
			txt_Occurence[num_tweet]['elong'] = techniques.countElongated(txt_English[num_tweet])
			#txt_Occurence[num_tweet]['spellDelta'] = []
			txt_Occurence[num_tweet]['user'] = txt_Import.at[num_tweet, 'user']
			txt_Occurence[num_tweet]['len'] = len(techniques.words(txt_English[num_tweet]))
			#print(txt_Occurence[num_tweet]['len'])

			spellDeltaASPELL = []
			spellDeltaLANGCHECK = []

			for word in i:
				if(word[1] is not ',' and word[1] is not 'U' and word[1] is not 'E' and word[1] is not 'G' and word[1] is not '$' and word[1] is not '#' and word[1] is not '@'):
					suggest = s.suggest(word[0])
					if(len(suggest) > 0):
						suggest = suggest[0]
					else:
						suggest = word[0]
					#print(word[0], word[1], suggest)
					#txt_Occurence[num_tweet]['spellDelta'].append(ls.distance(word[0], suggest))
					spellDeltaASPELL.append(ls.distance(word[0], suggest))

			fixedSentence = txt_English[num_tweet]
			sentenceMatches = lang_tool.check(fixedSentence)
			#print(sentenceMatches)
			language_check.correct(fixedSentence, sentenceMatches)
			num_word = 0
			#for words in fixedSentence.split(' '):
			for word in i:
				if(num_word < len(fixedSentence.split(' '))):
					#print(fixedSentence.split(' ')[num_word], txt_English[num_tweet].split(' ')[num_word])
					if(word[1] is not ',' and word[1] is not 'U' and word[1] is not 'E' and word[1] is not 'G' and word[1] is not '$' and word[1] is not '#' and [1] is not '@'):
					#print(word, txt_English[num_tweet].split(' ')[num_word], )
						spellDeltaLANGCHECK.append(ls.distance(fixedSentence.split(' ')[num_word], txt_English[num_tweet].split(' ')[num_word]))
					num_word = num_word + 1
			#print(txt_Occurence[num_tweet]['spellDelta'])

			#print(spellDeltaLANGCHECK, spellDeltaASPELL)

			txt_Occurence[num_tweet]['spellDelta'] = np.sum(np.add(spellDeltaLANGCHECK[:len(spellDeltaASPELL)], spellDeltaASPELL[:len(spellDeltaLANGCHECK)])/ ((len(spellDeltaASPELL) + len(spellDeltaLANGCHECK)) / 2))

			
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
	hashtag = sys.argv[1].split('.')[0]
	
	family = sys.argv[2]
	lang = sys.argv[3]
	print('Running '+ hashtag + ', '+family+', '+lang)
	result = analyzeText(hashtag, family, lang)
	print('Done')
	#print(result)