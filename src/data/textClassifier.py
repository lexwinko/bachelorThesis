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

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def classifyData(file, method, source):
	data = pd.read_csv(file, header=None, sep=',', skiprows=1)
	data.columns = ['correctedSentence', 'originalSentence', 'elongated','caps','sentenceLength','sentenceWordLength','spellDelta', 'charTrigrams','wordBigrams','wordUnigrams', 'hashtag', 'url', 'atUser','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M','langFam', 'lang', 'user',
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
					'wordUnigrams_similarity_English']
	data = data[data.correctedSentence.str.contains('correctedSentence') == False]
	data.sample(frac=1., random_state=42)
	classes = pd.get_dummies(pd.Series(list(data['lang'])))
	
	sentence = ['correctedSentence', 'originalSentence']

	if(source == 'reddit'):
		lang = [
			'French', 
			'German', 
			'Greek', 
			'English',  
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
						'wordUnigrams_similarity_English']
	else:
		lang = [
			'French', 
			'German', 
			'Greek', 
			'English', 
			'Indian', 
			'Japanese', 
			'Russian', 
			'Turkish', 
			]
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
						'wordUnigrams_similarity_English']

	le = preprocessing.LabelEncoder()
	le.fit(lang)
	print(le.classes_)

	featuredata = data[features].to_numpy(dtype='float64')
	labeldata = le.transform(data['lang'])

	X_train, X_test, y_train, y_test = train_test_split(featuredata, labeldata, test_size=0.3)
	

	model_results = {'Training':[], 'KFold':[]}

	if(method == 'tpot'):
		tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, cv=10, random_state=42)
		tpot.fit(X_train, y_train)
		print(tpot.score(X_test, y_test))
		tpot.export('tpot_model.py')
	else:
		exported_pipeline = make_pipeline(
	    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.7500000000000001, min_samples_leaf=17, min_samples_split=17, n_estimators=100, subsample=0.5)),
	    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=0.5, n_estimators=100), step=0.9000000000000001),
	    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.25, min_samples_leaf=14, min_samples_split=2, n_estimators=100)
		)

		exported_pipeline.fit(X_train, y_train)
		model_results['Training'] = {'prediction':exported_pipeline.predict(X_test), 'actual':y_test}


		kfold_splits = 10
		kf = KFold(n_splits=kfold_splits)
		for train, test in kf.split(X_train):
			exported_pipeline.fit(X_train[train], y_train[train])
			model_results['KFold'].append({'prediction':exported_pipeline.predict(X_train[test]), 'actual':y_train[test]})

		#print(model_results['GDC'][0],len(model_results['GDC']))
		#print(model_results['GDC'][0][0])
		#average =[sum(x[0])/len(model_results['GDC']) for x in model_results['GDC']]
		#print(average)
		#for model in model_results:
		#	print(model)
		#	average = 0
		#	average = [(average+value[0])/len(model_results[model]) for value in model_results[model]]
		#	print(sum(average))

		print('Training Accuracy Score: ',accuracy_score(model_results['Training']['actual'],model_results['Training']['prediction']))
		print('Training f1 macro Score: ',f1_score(model_results['Training']['actual'],model_results['Training']['prediction'],average='macro'))
		print('Training f1 micro Score: ',f1_score(model_results['Training']['actual'],model_results['Training']['prediction'],average='micro'))
		print('Training f1 weighted Score: ',f1_score(model_results['Training']['actual'],model_results['Training']['prediction'],average='weighted'))
		print('Training Precision Score: ',precision_score(model_results['Training']['actual'],model_results['Training']['prediction'],average='weighted'))
		print('Training Recall Score: ',recall_score(model_results['Training']['actual'],model_results['Training']['prediction'],average='weighted'))

		print('KFold Average Accuracy Score: '+str(sum([accuracy_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction']) for x in range(0,kfold_splits)])/kfold_splits))
		print('KFold f1 macro Score: '+str(sum([f1_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction'],average='macro') for x in range(0,kfold_splits)])/kfold_splits))
		print('KFold f1 micro Score: '+str(sum([f1_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction'],average='micro') for x in range(0,kfold_splits)])/kfold_splits))
		print('KFold f1 weighted Score: '+str(sum([f1_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
		print('KFold Precision Score: '+str(sum([precision_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
		print('KFold Recall Score: '+str(sum([recall_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction'],average='weighted') for x in range(0,kfold_splits)])/kfold_splits))
		#for entry in model_results['KFold']:
		#	print(accuracy_score(entry['actual'], entry['prediction']))
				
				




if __name__ == "__main__":
	data = sys.argv[1]
	method = sys.argv[2]
	source = sys.argv[3]
	print('Classifying '+source+' data: '+ data + ' with ' +method)
	classifyData(data, method, source)
	print('Done')