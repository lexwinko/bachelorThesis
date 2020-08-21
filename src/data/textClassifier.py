import pandas as pd
import numpy as np
import sys
import time
import csv

#import ktrain
#from ktrain import text as txt


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold
from sklearn import preprocessing
#from xgboost import XGBClassifier
#from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_selection import RFE
# from sklearn.pipeline import make_pipeline, make_union
# from sklearn.dummy import DummyClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.decomposition import IncrementalPCA
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.preprocessing import Normalizer
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from tpot.export_utils import set_param_recursive

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFwe, f_classif
from copy import copy


import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier


def classifyData(file, method, source, other):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence', 'elongated','caps','textLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'POSBigrams', 'functionWords', 'hashtag', 'url', '#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'category', 'origin',
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
	if(source == 'tfidf'):
		data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence', 'elongated','caps','textLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'POSBigrams', 'functionWords', 'hashtag', 'url', '#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'category', 'origin']
	else:
		data.columns = columns


	#data = data[data.correctedSentence.str.contains('correctedSentence') == False]
	#data.sample(random_state=42)
	#classes_family = pd.get_dummies(pd.Series(list(data['langFam'])))
	#classes_category = pd.get_dummies(pd.Series(list(data['category'])))
	#classes_lang = pd.get_dummies(pd.Series(list(data['lang'])))
	
	#sentence = ['correctedSentence', 'originalSentence', 'filteredSentence']

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
						#  '@',
						#  'E',
						',',
						#  '~',
						#  'U',
						'A',
						'D',
						'!',
						'N',
						'P',
						'O',
						'R',
						'&',
						#  'L',
						#  'Z',
						'^',
						'V',
						'$',
						'G',
						'T',
						#  'X',
						#  'S',
						#  'Y',
						#  'M',
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
						 #'@',
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
						# 'S',
						# 'Y',
						# 'M',
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
						#  '@',
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
						# 'S',
						#  'Y',
						#  'M',
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
	else:
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
						'M']

	#print(len(data))
	

	


	
	

	


	if(method == 'importance'):
		importance_data = {
							'Language' : {},
							'Language_Family' : {},
							'Category' : {},
							'Origin' : {},
		}

		featuredata = data[features].to_numpy(dtype='float64')

		le_family = preprocessing.LabelEncoder()
		le_family.fit(family)
		print(le_family.classes_)
		labeldata_family = le_family.transform(data['langFam'])

		le_category = preprocessing.LabelEncoder()
		le_category.fit(category)
		print(le_category.classes_)
		labeldata_category = le_category.transform(data['category'])

		le_lang = preprocessing.LabelEncoder()
		le_lang.fit(lang)
		print(le_lang.classes_)
		labeldata_lang = le_lang.transform(data['lang'])

		le_origin = preprocessing.LabelEncoder()
		le_origin.fit(origin)
		print(le_origin.classes_)
		labeldata_origin = le_origin.transform(data['origin'])



		classification_data = [labeldata_lang, labeldata_family, labeldata_category, labeldata_origin]
		labels = [lang, family, category, origin]

		classification_classes = ['Language', 'Language_Family', 'Category', 'Origin']
		for classes in range(0,4):
			if(source == 'reddit' or source == 'twitter') and (classification_classes[classes] == 'Category'):
				continue
			X_train, X_test, y_train, y_test = train_test_split(featuredata, classification_data[classes], test_size=0.3, random_state=42)



			model = RandomForestClassifier(criterion="gini", n_estimators=100,random_state=42)
			model.fit(X_train, y_train)
			#importance = model.feature_importances_
			feature_importance = model.feature_importances_
			importance = 100.0 * (feature_importance / feature_importance.max())
			#print(importance)
			importance_data[classification_classes[classes]] = pd.DataFrame({'importance':importance, 'feature':features})#, columns=['value']).sort_values(by=['value'],ascending=False)
			#for i,v in enumerate(importance):
			#	print('Feature: %0d, Score: %.5f' % (i,v))
			sorted_idx = np.argsort(importance)
			pos = np.arange(sorted_idx.shape[0]) + 0.9

			featfig = plt.figure(figsize=(7, 8))
			featax = featfig.add_subplot(1, 1, 1)
			featax.barh(pos, importance[sorted_idx], align='center')
			featax.set_yticks(pos)
			featax.set_yticklabels(np.array(features)[sorted_idx], fontsize=3.15)
			featax.set_xlabel('Relative Feature Importance '+classification_classes[classes])
			featax.set_xlim(0,max(importance))
			featax.set_ylim(bottom=0, top=(len(features)+0.9))

			plt.tight_layout()   
			plt.savefig('classification/importance_'+classification_classes[classes]+'_'+source, dpi=300, pad_inches=0)
			plt.close()

			sorted_idx = np.argsort(importance)[-10:]
			pos = np.arange(sorted_idx.shape[0]) + 0.5

			featfig = plt.figure(figsize=(4, 2))
			featax = featfig.add_subplot(1, 1, 1)
			featax.barh(pos, importance[sorted_idx], align='center')
			featax.set_yticks(pos)
			featax.set_yticklabels(np.array(features)[sorted_idx], fontsize=5)
			#featax.set_xlabel('Relative Feature Importance '+classification_classes[classes])
			featax.set_xlim(0,max(importance))
			featax.set_ylim(bottom=0, top=10)

			plt.tight_layout()   
			plt.savefig('classification/importance_'+classification_classes[classes]+'_'+source+'_top', dpi=300, pad_inches=0)
			plt.close()

			filename = 'classification/importance_report_'+classification_classes[classes]+'_'+source+'.csv'
			importance_data[classification_classes[classes]].to_csv(filename, header=[ 'RandomForestClassifier','Featurename'], index=False,float_format='%.5f')

		#fields = [ 'source', 'accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall']
		
		#importance_out = pd.DataFrame.from_dict(importance_data)
		#print(importance_out)
		#with open(filename, "a") as f:
		#	w = csv.DictWriter(f, fields)
		#	w.writeheader()
		#	w.writerow({'source': source, 'accuracy':acc_score, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'precision':pre_score, 'recall':rec_score})

		

		

	else:
		#tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range = (1,3), max_features=2000)
		#tfidf_matrix=tfidf_vectorizer.fit_transform(data['filteredSentence'].values.tolist())
		#dense = tfidf_matrix.todense()

		#featuredata = np.append(data[features].to_numpy(dtype='float64'),dense, axis=1)

		tpot = TPOTClassifier(generations=3, population_size=50, verbosity=2, cv=5, n_jobs=-1, random_state=42)

		
		chunks = createChunks(source,data,features,100)

		le_family = preprocessing.LabelEncoder()
		le_family.fit(family)
		print(le_family.classes_)
		labeldata_family = le_family.transform(data['langFam'])

		runs = 0
		scores = []
		best_score = {'score':0, 'run':0}
		while(sum([len(chunks[fam][lang]) for fam in [*chunks] for lang in [*chunks[fam]]])> 0):
			runs+=1
			print(runs,best_score)
			current_chunk = []
			for family in [*chunks]:
				for language in [*chunks[family]]:
					#print(language, len(chunks[family][language][0]),len(chunks[family][language][-1]))
					if(len(chunks[family][language]) > 0):
						#print(language)
						current_chunk.append(chunks[family][language][0])
						chunks[family][language].pop(0)
						#print(language, len(chunks[family][language]))
			#print(current_chunk)
			#print([frame.index for frame in current_chunk])
			current_chunk = pd.concat([frame for frame in current_chunk])


			#print(max(current_chunk.index))

			X_train, X_test, y_train, y_test = train_test_split(current_chunk, labeldata_family[(current_chunk.index-1)], test_size=0.3, random_state=42)

			#print(labeldata_family[(current_chunk.index)])

			tpot.fit(X_train, y_train)
			scores.append(tpot.score(X_test, y_test))
			if(max(scores) > best_score['score']):
				best_score['score'] = max(scores)
				best_score['run'] = runs
			elif(runs - best_score['run'] > 1):
				break

			


		tpot.export('tpot_model.py')

		


		#kfold_splits = 10
		#kf = KFold(n_splits=kfold_splits)
		#for train, test in kf.split(X_train):
		#	exported_pipeline.fit(X_train[train], y_train[train])
		#	dummy_clf.fit(X_train[train], y_train[train])
		#	model_results['TwitterPipeline']['KFold'].append({'prediction':exported_pipeline.predict(X_train[test]), 'actual':y_train[test]})
		#	model_results['Dummy']['KFold'].append({'prediction':dummy_clf.predict(X_train[test]), 'actual':y_train[test]})

		#print(model_results['GDC'][0],len(model_results['GDC']))
		#print(model_results['GDC'][0][0])
		#average =[sum(x[0])/len(model_results['GDC']) for x in model_results['GDC']]
		#print(average)
		#for model in model_results:
		#	print(model)
		#	average = 0
		#	average = [(average+value[0])/len(model_results[model]) for value in model_results[model]]
		#	print(sum(average))

		# for model in model_results:
		# 	print('Training Accuracy Score: ',accuracy_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction']))
		# 	print('Training f1 macro Score: ',f1_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='macro'))
		# 	print('Training f1 micro Score: ',f1_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='micro'))
		# 	print('Training f1 weighted Score: ',f1_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='weighted'))
		# 	print('Training Precision Score: ',precision_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='weighted'))
		# 	print('Training Recall Score: ',recall_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='weighted'))
		# 	cm = confusion_matrix(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'])

		# 	#print('KFold Average Accuracy Score: '+str(sum([accuracy_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction']) for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold f1 macro Score: '+str(sum([f1_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='macro') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold f1 micro Score: '+str(sum([f1_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='micro') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold f1 weighted Score: '+str(sum([f1_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold Precision Score: '+str(sum([precision_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold Recall Score: '+str(sum([recall_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))

		#for entry in model_results['KFold']:
		#	print(accuracy_score(entry['actual'], entry['prediction']))

		
def createChunks(source,dataset,featureset,csize):
	chunksize = csize
	family_groups = dataset.groupby(['langFam'], as_index=True)

	if(source == 'Combined' or source == 'CombinedNE' or source == 'Twitter'):

		BaltoSlavic = family_groups.get_group('Balto-Slavic').groupby(['lang'], as_index=True)[featureset]
		Germanic = family_groups.get_group('Germanic').groupby(['lang'], as_index=True)[featureset]
		Greek = family_groups.get_group('Greek')[featureset]
		IndoAryan = family_groups.get_group('Indo-Aryan')[featureset]
		Japonic = family_groups.get_group('Japonic')[featureset]
		Native = family_groups.get_group('Native')[featureset]
		Romance = family_groups.get_group('Romance').groupby(['lang'], as_index=True)[featureset]
		Turkic = family_groups.get_group('Turkic')[featureset]



		chunks = { 	'Balto-Slavic':{ 'Bulgarian':[],
									'Croatian': [],
									'Czech': [],
									'Lithuanian': [],
									'Polish': [],
									'Russian': [],
									'Serbian': [],
									'Slovene': []
									},
					'Germanic':{
									'German': [],
									'Finnish': [],
									'Dutch': [],
									'Norwegian':[],
									'Swedish' :[]
									},
					'Greek':{ 'Greek':[]},
					'Indo-Aryan':{ 'Indian': []},
					'Japonic':{ 'Japanese':[]},
					'Romance':{
									'French': [],
									'Italian' : [],
									'Spanish' :[],
									'Romanian': [],
									'Portuguese': []
								}, 
					'Turkic':{'Turkish':[]}, 
					'Native':{'Native':[]}
				}
		for g, df in Greek.groupby(np.arange(len(Greek)) // chunksize):
			chunks['Greek']['Greek'].append(df)

		for g, df in IndoAryan.groupby(np.arange(len(IndoAryan)) // chunksize):
			chunks['Indo-Aryan']['Indian'].append(df)

		for g, df in Turkic.groupby(np.arange(len(Turkic)) // chunksize):
			chunks['Turkic']['Turkish'].append(df)

		for g, df in Japonic.groupby(np.arange(len(Japonic)) // chunksize):
			chunks['Japonic']['Japanese'].append(df)
	else:

		BaltoSlavic = family_groups.get_group('Balto-Slavic').groupby(['lang'], as_index=True)[featureset]
		Germanic = family_groups.get_group('Germanic').groupby(['lang'], as_index=True)[featureset]
		Native = family_groups.get_group('Native')[featureset]
		Romance = family_groups.get_group('Romance').groupby(['lang'], as_index=True)[featureset]



		chunks = { 	'Balto-Slavic':{ 'Bulgarian':[],
									'Croatian': [],
									'Czech': [],
									'Lithuanian': [],
									'Polish': [],
									'Russian': [],
									'Serbian': [],
									'Slovene': []
									},
					'Germanic':{
									'German': [],
									'Finnish': [],
									'Dutch': [],
									'Norwegian':[],
									'Swedish' :[]
									},
					'Romance':{
									'French': [],
									'Italian' : [],
									'Spanish' :[],
									'Romanian': [],
									'Portuguese': []
								}, 
					'Native':{'Native':[]}
				}

	chunksize_bs = chunksize / max(1, np.count_nonzero([len(BaltoSlavic.get_group(language)) for language in BaltoSlavic.groups]))
	chunksize_gm = chunksize / max(1, np.count_nonzero([len(Germanic.get_group(language)) for language in Germanic.groups]))
	chunksize_ro = chunksize / max(1, np.count_nonzero([len(Romance.get_group(language)) for language in Romance.groups]))

	#print(chunksize_bs, chunksize_gm, chunksize_ro)


	for language in BaltoSlavic.groups:
		for g, df in BaltoSlavic.get_group(language).groupby(np.arange(len(BaltoSlavic.get_group(language))) // chunksize_bs):
		#for g, df in BaltoSlavic.get_group(language).groupby(np.arange(len(BaltoSlavic)) // chunksize):
			chunks['Balto-Slavic'][language].append(df)

	for language in Germanic.groups:
		for g, df in Germanic.get_group(language).groupby(np.arange(len(Germanic.get_group(language))) // chunksize_gm):
		#for g, df in Germanic.get_group(language).groupby(np.arange(len(Germanic)) // chunksize):
			chunks['Germanic'][language].append(df)

	

	for g, df in Native.groupby(np.arange(len(Native)) // chunksize):
		chunks['Native']['Native'].append(df)
		#if(g > 15):
		#	break

	for language in Romance.groups:
		for g, df in Romance.get_group(language).groupby(np.arange(len(Romance.get_group(language))) // chunksize_ro):
		#for g, df in Romance.get_group(language).groupby(np.arange(len(Romance)) // chunksize):
			chunks['Romance'][language].append(df)

	return chunks

def classifyDatasets(reddit, redditNE, twitter, combined, combinedNE):
	#datasets = [('Reddit',reddit), ('RedditNE',redditNE), ('Twitter',twitter), ('Combined',combined), ('CombinedNE',combinedNE)]
	datasets = [('RedditNE',redditNE)]
	class_scores = {
		'Normal':{ 
				'Reddit': {'Origin':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language_Family':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}},
				'RedditNE': {'Origin':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language_Family':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}},
				'Twitter': {'Origin':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language_Family':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}}
		},
		'TFIDF':{ 
				'Reddit': {'Origin':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language_Family':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}},
				'RedditNE': {'Origin':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language_Family':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}},
				'Twitter': {'Origin':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language_Family':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}, 'Language':{'RandomForest':[], 'Pipeline':[], 'LogisticRegression':[], 'SVM':[]}}
		}
	}
	for dataset in datasets:
		data = pd.read_csv(dataset[1], header=None, sep=',', skiprows=1)
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
		if(dataset[0] == 'Reddit' or dataset[0] == 'RedditNE'):

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
			
			features = [	'elongated',
							'caps',
							'textLength',
							'sentenceWordLength',
							'spellDelta',
							  '#',
							#  '@',
							#  'E',
							',',
							#  '~',
							#  'U',
							'A',
							'D',
							'!',
							'N',
							'P',
							'O',
							'R',
							'&',
							#  'L',
							#  'Z',
							'^',
							'V',
							'$',
							'G',
							'T',
							#  'X',
							#  'S',
							#  'Y',
							#  'M',
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
		elif(dataset[0] == 'Twitter'):
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

			features = [	'elongated',
							'caps',
							'textLength',
							'sentenceWordLength',
							'spellDelta',
							 '#',
							 #'@',
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
							# 'S',
							# 'Y',
							# 'M',
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
		else:
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
			features = [	'elongated',
							'caps',
							'textLength',
							'sentenceWordLength',
							'spellDelta',
							  '#',
							#  '@',
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
							# 'S',
							#  'Y',
							#  'M',
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


		


		#np.set_printoptions(threshold=sys.maxsize)
		le_family = preprocessing.LabelEncoder()
		le_family.fit(family)
		print(le_family.classes_)
		labeldata_family = le_family.transform(data['langFam'])
		le_name_mapping = dict(zip(le_family.classes_, le_family.transform(le_family.classes_)))
		print(le_name_mapping)

		le_category = preprocessing.LabelEncoder()
		le_category.fit(category)
		print(le_category.classes_)
		labeldata_category = le_category.transform(data['category'])
		le_name_mapping = dict(zip(le_category.classes_, le_category.transform(le_category.classes_)))
		print(le_name_mapping)

		le_lang = preprocessing.LabelEncoder()
		le_lang.fit(lang)
		print(le_lang.classes_)
		labeldata_lang = le_lang.transform(data['lang'])
		le_name_mapping = dict(zip(le_lang.classes_, le_lang.transform(le_lang.classes_)))
		print(le_name_mapping)

		le_origin = preprocessing.LabelEncoder()
		le_origin.fit(origin)
		print(le_origin.classes_)
		labeldata_origin = le_origin.transform(data['origin'])
		le_name_mapping = dict(zip(le_origin.classes_, le_origin.transform(le_origin.classes_)))
		print(le_name_mapping)



		classification_data = [labeldata_origin, labeldata_family, labeldata_lang]
		#classification_data = [labeldata_family, labeldata_lang]
		
		#labels = [lang, family, category, origin]

		if(dataset[0] == 'Combined' or dataset[0] == 'CombinedNE'):
			class_range = 4
		else:
			class_range = 3

		classification_classes = ['Origin', 'Language_Family', 'Language']
		#classification_classes = ['Language_Family', 'Language']
		
		for classes in range(0,class_range):
			print(dataset[0],classification_classes[classes])
			randomforest = RandomForestClassifier(criterion="gini", n_estimators=100,random_state=42, n_jobs=-1)

			svm = make_pipeline(StandardScaler(),LinearSVC(random_state=42, tol=1e-5))

			lr = LogisticRegression(random_state=42, n_jobs=-1)

			if(dataset[0] == 'twitter'):
				pipeline = make_pipeline(
				    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.3, min_samples_leaf=7, min_samples_split=6, n_estimators=100)),
				    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=18, min_samples_split=5, n_estimators=100)),
				    GaussianNB()
				)
				set_param_recursive(pipeline.steps, 'random_state', 42)
				set_param_recursive(pipeline.steps, 'n_jobs', -1)
			else:
				pipeline = LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)
				if hasattr(pipeline, 'random_state'):
				    setattr(pipeline, 'random_state', 42)
				if(hasattr(pipeline, 'n_jobs')):
					setattr(pipeline, 'n_jobs', -1)

			chunks = createChunks(dataset[0],data,features,100)


			while(sum([len(chunks[fam][lang]) for fam in [*chunks] for lang in [*chunks[fam]]]) > 0):
				print([len(chunks[fam][lang]) for fam in [*chunks] for lang in [*chunks[fam]]])
				current_chunk = []
				for family in [*chunks]:
					for language in [*chunks[family]]:
						#print(language, len(chunks[family][language][0]),len(chunks[family][language][-1]))
						if(len(chunks[family][language]) > 0):
							#print(language)
							current_chunk.append(chunks[family][language][0])
							chunks[family][language].pop(0)
							#print(language, len(chunks[family][language]))
				#print(current_chunk)
				#print([frame.index for frame in current_chunk])
				current_chunk = pd.concat([frame for frame in current_chunk])
				print(len(current_chunk))


				#print(max(current_chunk.index))
				#print(classification_data[classes], current_chunk.index)
				
				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					randomforest.fit(X_train, y_train)
					cv_resultsRF = cross_validate(randomforest, X_train, y_train, cv=5)
					predictionRF = randomforest.predict(X_test)
				except:
					print(current_chunk)
					#raise

				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					lr.fit(X_train, y_train)
					cv_resultsLR = cross_validate(lr, X_train, y_train, cv=5)
					predictionLR = lr.predict(X_test)
				except:
					print(current_chunk)
					#raise

				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					svm.fit(X_train, y_train)
					cv_resultsSVM = cross_validate(svm, X_train, y_train, cv=5)
					predictionSVM = svm.predict(X_test)
				except:
					print(current_chunk)
					#raise
				

				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					pipeline.fit(X_train, y_train)
					cv_resultsPipe = cross_validate(pipeline, X_train, y_train, cv=5)
					predictionPipe = pipeline.predict(X_test)
				except:
					print(current_chunk)
					#raise
				
				#print(class_scores['Normal'][dataset[0]][classification_classes[classes]]['RandomForest'])

				class_scores['Normal'][dataset[0]][classification_classes[classes]]['RandomForest'].append({'Prediction':predictionRF, 'Actual':y_test, 'CV':cv_resultsRF['test_score']})
				class_scores['Normal'][dataset[0]][classification_classes[classes]]['Pipeline'].append({'Prediction':predictionPipe, 'Actual':y_test, 'CV':cv_resultsPipe['test_score']})
				class_scores['Normal'][dataset[0]][classification_classes[classes]]['SVM'].append({'Prediction':predictionSVM, 'Actual':y_test, 'CV':cv_resultsSVM['test_score']})
				class_scores['Normal'][dataset[0]][classification_classes[classes]]['LogisticRegression'].append({'Prediction':predictionLR, 'Actual':y_test, 'CV':cv_resultsLR['test_score']})
				#print(class_scores['Normal'])
				#randomforest.n_estimators += 2

			#print(class_scores)
			#fig, ax = plt.subplots(figsize=(15, 15))
			#plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
			#plt.tight_layout()
			#plt.savefig('classification/'+run+'_All_Twitter', dpi=300)

			fields = [ 'accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'prediction', 'actual', 'cv']
			for model in class_scores['Normal'][dataset[0]][classification_classes[classes]]:
				for value in class_scores['Normal'][dataset[0]][classification_classes[classes]][model]:
					try:
						#print(dataset[0])
						acc_score = accuracy_score(value['Actual'],value['Prediction'])
						f1_macro = f1_score(value['Actual'],value['Prediction'],average='macro')
						f1_micro = f1_score(value['Actual'],value['Prediction'],average='micro')
						pre_score = precision_score(value['Actual'],value['Prediction'],average='weighted')
						rec_score = recall_score(value['Actual'],value['Prediction'],average='weighted')
						cv_mean = value['CV'].mean()
						#print(model+' '+source+' '+classification+' '+entry+' Accuracy Score: ',+acc_score)
						#print(model+' '+source+' '+classification+' '+entry+' f1 macro Score: ',+f1_macro)
						#print(model+' '+source+' '+classification+' '+entry+' f1 micro Score: ',+f1_micro)
						#print(model+' '+source+' '+classification+' '+entry+' Precision Score: ',+pre_score)
						#print(model+' '+source+' '+classification+' '+entry+' Recall Score: ',+rec_score)
						filename = 'classification/classification_report_'+dataset[0]+'_'+classification_classes[classes]+'_Normal_'+model+'.csv'
						with open(filename, "a") as f:
							w = csv.DictWriter(f, fields)
							w.writeheader()
							w.writerow({'accuracy':acc_score, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'precision':pre_score, 'recall':rec_score, 'prediction':value['Prediction'], 'actual':value['Actual'], 'cv':cv_mean})			
					except:
						print(value)
						raise



			features = [	'elongated',
							'caps',
							'textLength',
							'sentenceWordLength',
							'spellDelta',
							'#',
							#'@',
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
							'X'
							#'S',
							#'Y',
							#'M'
							]

			randomforest = RandomForestClassifier(criterion="gini", n_estimators=100,random_state=42, n_jobs=-1)

			svm = make_pipeline(StandardScaler(),LinearSVC(random_state=42, tol=1e-5))

			lr = LogisticRegression(random_state=42, n_jobs=-1)

			if(dataset[0] == 'twitter'):
				pipeline = make_pipeline(
				    StackingEstimator(estimator=RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.3, min_samples_leaf=7, min_samples_split=6, n_estimators=100)),
				    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=18, min_samples_split=5, n_estimators=100)),
				    GaussianNB()
				)
				set_param_recursive(pipeline.steps, 'random_state', 42)
				set_param_recursive(pipeline.steps, 'n_jobs', -1)
			else:
				pipeline = LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)
				if hasattr(pipeline, 'random_state'):
				    setattr(pipeline, 'random_state', 42)
				if(hasattr(pipeline, 'n_jobs')):
					setattr(pipeline, 'n_jobs', -1)

			tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range = (1,3), max_features=1500)
			tfidf_matrix=tfidf_vectorizer.fit_transform(data['stemmedSentence'].values.tolist())
			dense = tfidf_matrix.todense()
			tfidfdata = pd.concat([data,pd.DataFrame(dense)],axis=1).fillna(0)



			chunks = createChunks(dataset[0],tfidfdata,list(set(features) | (set(tfidfdata.columns) - set(data.columns))),100)

			

			while(sum([len(chunks[fam][lang]) for fam in [*chunks] for lang in [*chunks[fam]]])> 0):
				print([len(chunks[fam][lang]) for fam in [*chunks] for lang in [*chunks[fam]]])
				current_chunk = []
				for family in [*chunks]:
					for language in [*chunks[family]]:
						#print(language, len(chunks[family][language][0]),len(chunks[family][language][-1]))
						if(len(chunks[family][language]) > 0):
							#print(language)
							#np.append(chunks[family][language][0].to_numpy(dtype='float64'),dense, axis=1)
							current_chunk.append(chunks[family][language][0])
							chunks[family][language].pop(0)
							#print(language, len(chunks[family][language]))
				#print(current_chunk)
				#print([frame.index for frame in current_chunk])
				#print([ch for ch in current_chunk])
				current_chunk = pd.concat([frame for frame in current_chunk])

				#print(current_chunk)


				#print(max(current_chunk.index))
				#print(classification_data[classes], current_chunk.index)
				
				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					randomforest.fit(X_train, y_train)
					cv_resultsRF = cross_validate(randomforest, X_train, y_train, cv=5)
					predictionRF = randomforest.predict(X_test)
				except:
					print(current_chunk)

				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					lr.fit(X_train, y_train)
					cv_resultsLR = cross_validate(lr, X_train, y_train, cv=5)
					predictionLR = lr.predict(X_test)
				except:
					print(current_chunk)

				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					svm.fit(X_train, y_train)
					cv_resultsSVM = cross_validate(svm, X_train, y_train, cv=5)
					predictionSVM = svm.predict(X_test)
				except:
					print(current_chunk)
				

				X_train, X_test, y_train, y_test = train_test_split(current_chunk, classification_data[classes][(current_chunk.index)], test_size=0.3, random_state=42)
				try:
					pipeline.fit(X_train, y_train)
					cv_resultsPipe = cross_validate(pipeline, X_train, y_train, cv=5)
					predictionPipe = pipeline.predict(X_test)
				except:
					print(current_chunk)
				
				#print(class_scores['Normal'][dataset[0]][classification_classes[classes]]['RandomForest'])

				class_scores['TFIDF'][dataset[0]][classification_classes[classes]]['RandomForest'].append({'Prediction':predictionRF, 'Actual':y_test, 'CV':cv_resultsRF['test_score']})
				class_scores['TFIDF'][dataset[0]][classification_classes[classes]]['Pipeline'].append({'Prediction':predictionPipe, 'Actual':y_test, 'CV':cv_resultsPipe['test_score']})
				class_scores['TFIDF'][dataset[0]][classification_classes[classes]]['SVM'].append({'Prediction':predictionSVM, 'Actual':y_test, 'CV':cv_resultsSVM['test_score']})
				class_scores['TFIDF'][dataset[0]][classification_classes[classes]]['LogisticRegression'].append({'Prediction':predictionLR, 'Actual':y_test, 'CV':cv_resultsLR['test_score']})
				#print(class_scores['Normal'])
				#randomforest.n_estimators += 2

			#print(class_scores)
			#fig, ax = plt.subplots(figsize=(15, 15))
			#plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
			#plt.tight_layout()
			#plt.savefig('classification/'+run+'_All_Twitter', dpi=300)

			fields = [ 'accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'prediction', 'actual', 'cv']
			for model in class_scores['TFIDF'][dataset[0]][classification_classes[classes]]:
				for value in class_scores['TFIDF'][dataset[0]][classification_classes[classes]][model]:
					try:
						#print(dataset[0])
						acc_score = accuracy_score(value['Actual'],value['Prediction'])
						f1_macro = f1_score(value['Actual'],value['Prediction'],average='macro')
						f1_micro = f1_score(value['Actual'],value['Prediction'],average='micro')
						pre_score = precision_score(value['Actual'],value['Prediction'],average='weighted')
						rec_score = recall_score(value['Actual'],value['Prediction'],average='weighted')
						cv_mean = value['CV'].mean()
						#print(model+' '+source+' '+classification+' '+entry+' Accuracy Score: ',+acc_score)
						#print(model+' '+source+' '+classification+' '+entry+' f1 macro Score: ',+f1_macro)
						#print(model+' '+source+' '+classification+' '+entry+' f1 micro Score: ',+f1_micro)
						#print(model+' '+source+' '+classification+' '+entry+' Precision Score: ',+pre_score)
						#print(model+' '+source+' '+classification+' '+entry+' Recall Score: ',+rec_score)
						filename = 'classification/classification_report_'+dataset[0]+'_'+classification_classes[classes]+'_TFIDF_'+model+'.csv'
						with open(filename, "a") as f:
							w = csv.DictWriter(f, fields)
							w.writeheader()
							w.writerow({'accuracy':acc_score, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'precision':pre_score, 'recall':rec_score, 'prediction':value['Prediction'], 'actual':value['Actual'], 'cv':cv_mean})			
					except:
						print(value)
						raise

		# fields = [ 'accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall', 'prediction', 'actual']
		# for model in class_scores:
		# 	for classification in class_scores[model][dataset[0]]:
		# 		for entry in class_scores[model][dataset[0]][classification]:
		# 			for value in class_scores[model][dataset[0]][classification][entry]:
		# 				try:
		# 					print(value)
		# 					acc_score = accuracy_score(value['Actual'],value['Prediction'])
		# 					f1_macro = f1_score(value['Actual'],value['Prediction'],average='macro')
		# 					f1_micro = f1_score(value['Actual'],value['Prediction'],average='micro')
		# 					pre_score = precision_score(value['Actual'],value['Prediction'],average='weighted')
		# 					rec_score = recall_score(value['Actual'],value['Prediction'],average='weighted')
		# 					#print(model+' '+source+' '+classification+' '+entry+' Accuracy Score: ',+acc_score)
		# 					#print(model+' '+source+' '+classification+' '+entry+' f1 macro Score: ',+f1_macro)
		# 					#print(model+' '+source+' '+classification+' '+entry+' f1 micro Score: ',+f1_micro)
		# 					#print(model+' '+source+' '+classification+' '+entry+' Precision Score: ',+pre_score)
		# 					#print(model+' '+source+' '+classification+' '+entry+' Recall Score: ',+rec_score)
		# 					filename = 'classification/classification_report_'+dataset[0]+'_'+classification+'_'+model+'_'+entry+'.csv'
		# 					with open(filename, "a") as f:
		# 						w = csv.DictWriter(f, fields)
		# 						w.writeheader()
		# 						w.writerow({'accuracy':acc_score, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'precision':pre_score, 'recall':rec_score, 'prediction':value['Prediction'], 'actual':value['Actual'], 'cv':value['CV']})			
		# 				except:
		# 					print(value)
if __name__ == "__main__":
	data = sys.argv[1]
	method = sys.argv[2]
	source = sys.argv[3]
	other = sys.argv[4]
	print('Classifying '+source+' data: '+ data + ' with ' +method)
	#classifyData(data, method, source, other)
	classifyDatasets(data, method, source, other, sys.argv[5])
	print('Done')