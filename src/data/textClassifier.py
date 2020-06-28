import pandas as pd
import numpy as np
import sys
import time
import csv

#import ktrain
#from ktrain import text as txt


from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import preprocessing
from xgboost import XGBClassifier
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline, make_union
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFwe, f_classif
from copy import copy
from sklearn.preprocessing import Binarizer

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier


model_results = {		'Reddit_All':{'Training':[], 'Vectorized':[]},
						'Twitter_All':{'Training':[], 'Vectorized':[]},
						'Combined_All':{'Training':[], 'Vectorized':[]},
						'Reddit_Importance':{'Training':[], 'Vectorized':[]},
						'Twitter_Importance':{'Training':[], 'Vectorized':[]},
						'Combined_Importance':{'Training':[], 'Vectorized':[]},
						'Reddit_tfidf':{'Training':[], 'Vectorized':[]},
						'Twitter_tfidf':{'Training':[], 'Vectorized':[]},
						'Combined_tfidf':{'Training':[], 'Vectorized':[]}}



def classifyData(file, method, source, other):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	if(source == 'tfidf'):
		data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'user', 'category']
	else:
		data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence','stemmedSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'user', 'category',
						'charTrigrams_similarity_French',
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


	data = data[data.correctedSentence.str.contains('correctedSentence') == False]
	data.sample(random_state=42)
	#classes_family = pd.get_dummies(pd.Series(list(data['langFam'])))
	#classes_category = pd.get_dummies(pd.Series(list(data['category'])))
	#classes_lang = pd.get_dummies(pd.Series(list(data['lang'])))
	
	sentence = ['correctedSentence', 'originalSentence', 'filteredSentence']

	if(other == 'reddit'):
		lang = [
			'Bulgarian',
			'Croatian',
			'Czech',
			'Dutch',
			'English',
			'Estonian',
			'Finnish',
			'French', 
			'German', 
			'Greek', 
			'Hungarian',
			'Italian',
			'Lithuanian',
			'Norwegian',
			'Polish',
			'Portugese',
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
				'Romance',
				'Turkic',
				'Uralic'
		]
		category = [
				'European',
				'NonEuropean',
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
		]
		category = [
				'ArtCul',
				'BuiTecSci',
				'Pol',
				'SocSoc'
		]
	elif(other == 'combined'):
		lang = [
			'Bulgarian',
			'Croatian',
			'Czech',
			'Dutch',
			'English',
			'Estonian',
			'Finnish',
			'French', 
			'German', 
			'Greek', 
			'Hungarian',
			'Indian',
			'Italian',
			'Japanese',
			'Lithuanian',
			'Norwegian',
			'Polish',
			'Portugese',
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
				'Uralic'
		]
		category = [
				'ArtCul',
				'BuiTecSci',
				'European',
				'NonEuropean',
				'Pol',
				'SocSoc'
		]
	

	if(source == 'reddit'):
		
		features = [	'elongated',
						'caps',
						'sentenceLength',
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
						  'charTrigrams_similarity_German',
						'wordBigrams_similarity_German',
						'wordUnigrams_similarity_German',
						  'charTrigrams_similarity_Greek',
						'wordBigrams_similarity_Greek',
						'wordUnigrams_similarity_Greek',
						  'charTrigrams_similarity_Russian',
						'wordBigrams_similarity_Russian',
						'wordUnigrams_similarity_Russian',
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
						  'charTrigrams_similarity_Turkic',
						'wordBigrams_similarity_Turkic',
						'wordUnigrams_similarity_Turkic',
						  'charTrigrams_similarity_Uralic',
						'wordBigrams_similarity_Uralic',
						'wordUnigrams_similarity_Uralic',
						  'charTrigrams_similarity_European',
						'wordBigrams_similarity_European',
						'wordUnigrams_similarity_European',
						  'charTrigrams_similarity_NonEuropean',
						'wordBigrams_similarity_NonEuropean',
						'wordUnigrams_similarity_NonEuropean']
	elif(source == 'twitter'):
		
		features = [	'elongated',
						'caps',
						'sentenceLength',
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
						 'charTrigrams_similarity_Turkic',
						'wordBigrams_similarity_Turkic',
						'wordUnigrams_similarity_Turkic',
						 'charTrigrams_similarity_Indo-Aryan',
						'wordBigrams_similarity_Indo-Aryan',
						'wordUnigrams_similarity_Indo-Aryan',
						 'charTrigrams_similarity_Japonic',
						'wordBigrams_similarity_Japonic',
						'wordUnigrams_similarity_Japonic',
						 'charTrigrams_similarity_ArtCul',
						'wordBigrams_similarity_ArtCul',
						'wordUnigrams_similarity_ArtCul',
						 'charTrigrams_similarity_BuiTecSci',
						'wordBigrams_similarity_BuiTecSci',
						'wordUnigrams_similarity_BuiTecSci',
						 'charTrigrams_similarity_SocSoc',
						'wordBigrams_similarity_SocSoc',
						'wordUnigrams_similarity_SocSoc',
						 'charTrigrams_similarity_Pol',
						'wordBigrams_similarity_Pol',
						'wordUnigrams_similarity_Pol']
	elif(source == 'combined'):
		
		features = [	'elongated',
						'caps',
						'sentenceLength',
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
						'wordUnigrams_similarity_English',
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
						  'charTrigrams_similarity_Turkic',
						'wordBigrams_similarity_Turkic',
						'wordUnigrams_similarity_Turkic',
						  'charTrigrams_similarity_Uralic',
						'wordBigrams_similarity_Uralic',
						'wordUnigrams_similarity_Uralic',
						  'charTrigrams_similarity_Indo-Aryan',
						'wordBigrams_similarity_Indo-Aryan',
						'wordUnigrams_similarity_Indo-Aryan',
						  'charTrigrams_similarity_Japonic',
						'wordBigrams_similarity_Japonic',
						'wordUnigrams_similarity_Japonic',
						  'charTrigrams_similarity_ArtCul',
						'wordBigrams_similarity_ArtCul',
						'wordUnigrams_similarity_ArtCul',
						  'charTrigrams_similarity_BuiTecSci',
						'wordBigrams_similarity_BuiTecSci',
						'wordUnigrams_similarity_BuiTecSci',
						  'charTrigrams_similarity_SocSoc',
						'wordBigrams_similarity_SocSoc',
						'wordUnigrams_similarity_SocSoc',
						  'charTrigrams_similarity_Pol',
						'wordBigrams_similarity_Pol',
						'wordUnigrams_similarity_Pol',
						  'charTrigrams_similarity_European',
						'wordBigrams_similarity_European',
						'wordUnigrams_similarity_European',
						  'charTrigrams_similarity_NonEuropean',
						'wordBigrams_similarity_NonEuropean',
						'wordUnigrams_similarity_NonEuropean']
	else:
		features = [	'elongated',
						'caps',
						'sentenceLength',
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

	featuredata = data[features].to_numpy(dtype='float64')

	classification_data = labeldata_lang
	labels = lang


	X_train, X_test, y_train, y_test = train_test_split(featuredata, classification_data, test_size=0.3)



	

	if(method == 'classify'):
		runClassifiers(X_train, y_train, X_test, y_test, labels, 'Training')


		features = [	'elongated',
						'caps',
						'sentenceLength',
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



		tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range = (1,3), max_features=5000)
		tfidf_matrix=tfidf_vectorizer.fit_transform(data['filteredSentence'].values.tolist())
		dense = tfidf_matrix.todense()
		featuredata = np.append(data[features].to_numpy(dtype='float64'),dense, axis=1)
		X_train, X_test, y_train, y_test = train_test_split(featuredata, classification_data, test_size=0.3)

		runClassifiers(X_train, y_train, X_test, y_test, labels, 'Vectorized')



		fields = [ 'source', 'accuracy', 'f1_macro', 'f1_micro', 'precision', 'recall']
		for model in model_results:
			print(model)
			acc_score = 0
			f1_macro = 0
			f1_micro = 0
			pre_score = 0
			rec_score = 0
			for version in model_results[model]:
				print(version)
				print(model_results[model][version]['actual'],model_results[model][version]['prediction'])
				acc_score = accuracy_score(model_results[model][version]['actual'],model_results[model][version]['prediction'])
				f1_macro = f1_score(model_results[model][version]['actual'],model_results[model][version]['prediction'],average='macro')
				f1_micro = f1_score(model_results[model][version]['actual'],model_results[model][version]['prediction'],average='micro')
				pre_score = precision_score(model_results[model][version]['actual'],model_results[model][version]['prediction'],average='weighted')
				rec_score = recall_score(model_results[model][version]['actual'],model_results[model][version]['prediction'],average='weighted')
				print(model+' '+version+' Accuracy Score: ',+acc_score)
				print(model+' '+version+' f1 macro Score: ',+f1_macro)
				print(model+' '+version+' f1 micro Score: ',+f1_micro)
				print(model+' '+version+' Precision Score: ',+pre_score)
				print(model+' '+version+' Recall Score: ',+rec_score)
				print('')
				filename = 'classification/classification_report_'+model+'_'+version+'.csv'
				with open(filename, "a") as f:
					w = csv.DictWriter(f, fields)
					w.writeheader()
					w.writerow({'source': source, 'accuracy':acc_score, 'f1_macro':f1_macro, 'f1_micro':f1_micro, 'precision':pre_score, 'recall':rec_score})



	elif(method == 'importance'):

		model = make_pipeline(
		StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.7500000000000001, min_samples_leaf=17, min_samples_split=17, n_estimators=100, subsample=0.5)),
		RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.5, n_estimators=100), step=0.9000000000000001),
		RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.25, min_samples_leaf=14, min_samples_split=2, n_estimators=100)
		)
		model.fit(X_train, y_train)
		importance = model.feature_importances_
		for i,v in enumerate(importance):
			print('Feature: %0d, Score: %.5f' % (i,v))
		plt.bar([features[x] for x in range(len(importance))], importance)
		plt.tight_layout()
		plt.xticks(rotation=90)
		plt.show()

		model = make_pipeline(
		StackingEstimator(estimator=LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)),
		StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=11, min_samples_split=6)),
		RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
		)
		# Fix random state for all the steps in exported pipeline
		set_param_recursive(model.steps, 'random_state', 42)
		model.fit(X_train, y_train)
		importance = model.feature_importances_
		for i,v in enumerate(importance):
			print('Feature: %0d, Score: %.5f' % (i,v))
		plt.bar([features[x] for x in range(len(importance))], importance)
		plt.tight_layout()
		plt.xticks(rotation=90)
		plt.show()

	else:
		tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range = (1,3), max_features=2000)
		tfidf_matrix=tfidf_vectorizer.fit_transform(data['filteredSentence'].values.tolist())
		dense = tfidf_matrix.todense()

		featuredata = np.append(data[features].to_numpy(dtype='float64'),dense, axis=1)

		X_train, X_test, y_train, y_test = train_test_split(featuredata, labeldata_family, test_size=0.3)

		tpot = TPOTClassifier(generations=4, population_size=30, verbosity=2, cv=10, random_state=42)
		tpot.fit(X_train, y_train)
		print(tpot.score(X_test, y_test))
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


			

def runClassifiers(X_test, y_test, X_train, y_train, labeldata, run='Training'):
	#########	TWITTER MODELS
	# Average CV score on the training set was: 0.7985221674876848
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.15000000000000002, min_samples_leaf=8, min_samples_split=5, n_estimators=100)),
	    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
	)
	# Fix random state for all the steps in exported pipeline
	set_param_recursive(exported_pipeline.steps, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_All_Twitter', dpi=300)
	model_results['Twitter_All'][run] = {'prediction':results, 'actual':y_test}

	# Average CV score on the training set was: 0.7655172413793103
	exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=7, min_samples_leaf=20, min_samples_split=16)),
	    MinMaxScaler(),
	    StackingEstimator(estimator=LinearSVC(C=1.0, dual=True, loss="squared_hinge", penalty="l2", tol=0.001)),
	    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.7000000000000001, min_samples_leaf=6, min_samples_split=9, n_estimators=100)
	)
	# Fix random state for all the steps in exported pipeline
	set_param_recursive(exported_pipeline.steps, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_Importance_Twitter', dpi=300)
	model_results['Twitter_Importance'][run] = {'prediction':results, 'actual':y_test}

	# Average CV score on the training set was: 0.6083743842364532
	exported_pipeline = make_pipeline(
	    make_union(
	        FunctionTransformer(copy),
	        FunctionTransformer(copy)
	    ),
	    LinearSVC(C=0.5, dual=False, loss="squared_hinge", penalty="l1", tol=0.01)
	)
	# Fix random state for all the steps in exported pipeline
	set_param_recursive(exported_pipeline.steps, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_tfidf_Twitter', dpi=300)
	model_results['Twitter_tfidf'][run] = {'prediction':results, 'actual':y_test}

	#########	REDDIT MODELS

	# Average CV score on the training set was: 0.39737073599843614
	exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.05, min_samples_leaf=7, min_samples_split=11, n_estimators=100, subsample=0.6000000000000001)
	# Fix random state in exported estimator
	if hasattr(exported_pipeline, 'random_state'):
	    setattr(exported_pipeline, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_All_Reddit', dpi=300)
	model_results['Reddit_All'][run] = {'prediction':results, 'actual':y_test}

	# Average CV score on the training set was: 0.2268686725080395
	exported_pipeline = make_pipeline(
	    Normalizer(norm="max"),
	    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.7000000000000001, min_samples_leaf=12, min_samples_split=18, n_estimators=100)
	)
	# Fix random state for all the steps in exported pipeline
	set_param_recursive(exported_pipeline.steps, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_Importance_Reddit', dpi=300)
	model_results['Reddit_Importance'][run] = {'prediction':results, 'actual':y_test}

	# Average CV score on the training set was: 0.4016895256758951
	exported_pipeline = LinearSVC(C=0.5, dual=False, loss="squared_hinge", penalty="l2", tol=0.001)
	# Fix random state in exported estimator
	if hasattr(exported_pipeline, 'random_state'):
	    setattr(exported_pipeline, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_tfidf_Reddit', dpi=300)
	model_results['Reddit_tfidf'][run] = {'prediction':results, 'actual':y_test}
	

	#########	COMBINED MODELS
	# Average CV score on the training set was: 0.4315001709872857
	exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=4, min_samples_split=13)
	# Fix random state in exported estimator
	if hasattr(exported_pipeline, 'random_state'):
	    setattr(exported_pipeline, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_All_Combined', dpi=300)
	model_results['Combined_All'][run] = {'prediction':results, 'actual':y_test}

	# Average CV score on the training set was: 0.43276417535639367
	exported_pipeline = DecisionTreeClassifier(criterion="entropy", max_depth=8, min_samples_leaf=19, min_samples_split=18)
	# Fix random state in exported estimator
	if hasattr(exported_pipeline, 'random_state'):
	    setattr(exported_pipeline, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	results = exported_pipeline.predict(X_test)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_Importance_Combined', dpi=300)
	model_results['Combined_Importance'][run] = {'prediction':results, 'actual':y_test}

	# Average CV score on the training set was: 0.41023381873306075
	exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=9, min_samples_leaf=19, min_samples_split=12)
	# Fix random state in exported estimator
	if hasattr(exported_pipeline, 'random_state'):
	    setattr(exported_pipeline, 'random_state', 42)

	exported_pipeline.fit(X_train, y_train)
	fig, ax = plt.subplots(figsize=(15, 15))
	plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=labeldata,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues, ax=ax)
	plt.tight_layout()
	plt.savefig('classification/'+run+'_tfidf_Combined', dpi=300)
	model_results['Combined_tfidf'][run] = {'prediction':results, 'actual':y_test}


if __name__ == "__main__":
	data = sys.argv[1]
	method = sys.argv[2]
	source = sys.argv[3]
	other = sys.argv[4]
	print('Classifying '+source+' data: '+ data + ' with ' +method)
	classifyData(data, method, source, other)
	print('Done')