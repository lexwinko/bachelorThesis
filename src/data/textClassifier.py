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
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline, make_union


def classifyData(file, method):
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
					'charTrigrams_similarity_English',
					'wordBigrams_similarity_English',
					'wordUnigrams_similarity_English']
	data = data[data.correctedSentence.str.contains('correctedSentence') == False]
	data.sample(frac=1., random_state=int(time.time()))
	classes = pd.get_dummies(pd.Series(list(data['lang'])))
	lang = ['French', 'German', 'Greek', 'English', 'Indian', 'Japanese', 'Russian', 'Turkish']
	sentence = ['correctedSentence', 'originalSentence']
	
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
	#tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)

	

	


	#tpot.fit(X_train, y_train)
	#print(tpot.score(X_test, y_test))
	#tpot.export('tpot_model.py')

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

	print('Training Accuracy: ',accuracy_score(model_results['Training']['actual'],model_results['Training']['prediction']))

	print('KFold Average Accuracy: '+str(sum([accuracy_score(model_results['KFold'][x]['actual'],model_results['KFold'][x]['prediction']) for x in range(0,kfold_splits)])/kfold_splits))
	for entry in model_results['KFold']:
		print(accuracy_score(entry['actual'], entry['prediction']))
			
			




if __name__ == "__main__":
	data = sys.argv[1]
	if(len(sys.argv) > 2):
		method = sys.argv[2]
		print('Classifying '+ data + ' with ' +method)
		classifyData(data, method)
	else:
		print('Classifying '+ data +' with TPOT')
		classifyData(data, 'none')
	print('Done')