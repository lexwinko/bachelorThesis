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
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline, make_union
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import IncrementalPCA

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


def classifyData(file, method, source):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	data.columns = ['correctedSentence', 'originalSentence', 'filteredSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'user', 'category',
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
	classes_family = pd.get_dummies(pd.Series(list(data['langFam'])))
	classes_category = pd.get_dummies(pd.Series(list(data['category'])))
	classes_lang = pd.get_dummies(pd.Series(list(data['lang'])))
	
	sentence = ['correctedSentence', 'originalSentence', 'filteredSentence']

	lang = [
			'French', 
			'German', 
			'Greek', 
			'English',  
			'Russian', 
			'Turkish',
			'Japanese',
			'Indian',
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
	family = [
			'Balto-Slavic',
			'Germanic',
			'Indo-Aryan',
			'Japonic',
			'Romance',
			'Turkic',
			'Uralic',
			'Greek'
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

	X_train, X_test, y_train, y_test = train_test_split(featuredata, labeldata_family, test_size=0.3)
	

	model_results = {	'Dummy':{'Training':[], 'KFold':[]},
						'DTC':{'Training':[], 'KFold':[]},
						'RFC':{'Training':[], 'KFold':[]},
						'XGB':{'Training':[], 'KFold':[]},
					 	'TwitterPipeline':{'Training':[], 'KFold':[]},
					 	'TwitterPipeline2':{'Training':[], 'KFold':[]}}

	if(method == 'tpot'):
		tpot = TPOTClassifier(generations=3, population_size=20, verbosity=2, cv=10, random_state=42)
		tpot.fit(X_train, y_train)
		print(tpot.score(X_test, y_test))
		tpot.export('tpot_model.py')
	elif(method == 'importance'):

		model = DecisionTreeClassifier()
		model.fit(X_train, y_train)
		importance = model.feature_importances_
		for i,v in enumerate(importance):
			print('Feature: %0d, Score: %.5f' % (i,v))
		plt.bar([features[x] for x in range(len(importance))], importance)
		plt.tight_layout()
		plt.xticks(rotation=90)
		plt.show()

		model = RandomForestClassifier()
		model.fit(X_train, y_train)
		importance = model.feature_importances_
		for i,v in enumerate(importance):
			print('Feature: %0d, Score: %.5f' % (i,v))
		plt.bar([features[x] for x in range(len(importance))], importance)
		plt.tight_layout()
		plt.xticks(rotation=90)
		plt.show()

		model = XGBClassifier()
		model.fit(X_train, y_train)
		importance = model.feature_importances_
		for i,v in enumerate(importance):
			print('Feature: %0d, Score: %.5f' % (i,v))
		plt.bar([features[x] for x in range(len(importance))], importance)
		plt.tight_layout()
		plt.xticks(rotation=90)
		plt.show()
	else:
		exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.7500000000000001, min_samples_leaf=17, min_samples_split=17, n_estimators=100, subsample=0.5)),
	    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.5, n_estimators=100), step=0.9000000000000001),
	    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.25, min_samples_leaf=14, min_samples_split=2, n_estimators=100)
		)

		exported_pipeline.fit(X_train, y_train)
		plot_confusion_matrix(exported_pipeline, X_test, y_test, display_labels=lang,normalize=None,  values_format='d', xticks_rotation='vertical',cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('pipeline_'+source, dpi=300)
		model_results['TwitterPipeline']['Training'] = {'prediction':exported_pipeline.predict(X_test), 'actual':y_test}

		# Dummy Most Frequent
		dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
		dummy_clf.fit(X_train, y_train)
		plot_confusion_matrix(dummy_clf,X_test, y_test, display_labels=lang,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('dummyStratified_'+source, dpi=300)
		model_results['Dummy']['Training'] = {'prediction':dummy_clf.predict(X_test), 'actual':y_test}

		dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
		dummy_clf.fit(X_train, y_train)
		plot_confusion_matrix(dummy_clf,X_test, y_test, display_labels=lang,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('dummyMF_'+source, dpi=300)


		model = DecisionTreeClassifier()
		model.fit(X_train, y_train)
		plot_confusion_matrix(model,X_test, y_test, display_labels=lang,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('DTC_'+source, dpi=300)
		model_results['DTC']['Training'] = {'prediction':model.predict(X_test), 'actual':y_test}


		model = RandomForestClassifier()
		model.fit(X_train, y_train)
		plot_confusion_matrix(model,X_test, y_test, display_labels=lang,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('RFC_'+source, dpi=300)
		model_results['RFC']['Training'] = {'prediction':model.predict(X_test), 'actual':y_test}


		model = XGBClassifier()
		model.fit(X_train, y_train)
		plot_confusion_matrix(model,X_test, y_test, display_labels=lang,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('XGB_'+source, dpi=300)
		model_results['XGB']['Training'] = {'prediction':model.predict(X_test), 'actual':y_test}

		exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=LinearSVC(C=10.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)),
	    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=11, min_samples_split=6)),
	    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=8, n_estimators=100)
		)
		# Fix random state for all the steps in exported pipeline
		set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		exported_pipeline.fit(X_train, y_train)
		plot_confusion_matrix(exported_pipeline,X_test, y_test, display_labels=lang,normalize=None,values_format='d', xticks_rotation='vertical', cmap=plt.cm.Blues)
		plt.tight_layout()
		plt.savefig('pipeline2_'+source, dpi=300)
		model_results['TwitterPipeline2']['Training'] = {'prediction':exported_pipeline.predict(X_test), 'actual':y_test}



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

		for model in model_results:
			print('Training Accuracy Score: ',accuracy_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction']))
			print('Training f1 macro Score: ',f1_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='macro'))
			print('Training f1 micro Score: ',f1_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='micro'))
			print('Training f1 weighted Score: ',f1_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='weighted'))
			print('Training Precision Score: ',precision_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='weighted'))
			print('Training Recall Score: ',recall_score(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'],average='weighted'))
			cm = confusion_matrix(model_results[model]['Training']['actual'],model_results[model]['Training']['prediction'])

			#print('KFold Average Accuracy Score: '+str(sum([accuracy_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction']) for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold f1 macro Score: '+str(sum([f1_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='macro') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold f1 micro Score: '+str(sum([f1_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='micro') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold f1 weighted Score: '+str(sum([f1_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold Precision Score: '+str(sum([precision_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
			#print('KFold Recall Score: '+str(sum([recall_score(model_results[model]['KFold'][x]['actual'],model_results[model]['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))

		#for entry in model_results['KFold']:
		#	print(accuracy_score(entry['actual'], entry['prediction']))
				
				




if __name__ == "__main__":
	data = sys.argv[1]
	method = sys.argv[2]
	source = sys.argv[3]
	print('Classifying '+source+' data: '+ data + ' with ' +method)
	classifyData(data, method, source)
	print('Done')