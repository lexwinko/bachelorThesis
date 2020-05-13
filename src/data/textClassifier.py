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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import FastICA
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.metrics import accuracy_score


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
	

	model_results = {'GDC':[], 'SGD':[], 'ICA':[], 'XGB':[]}
	tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)

	kf = KFold(n_splits=5)
	split_nr = 1
	for train, test in kf.split(featuredata):
		print(len(train),len(test))

		tpot.fit(featuredata[train], labeldata[train])
		print(tpot.score(featuredata[test], labeldata[test]))
		tpot.export('tpot_ksplit_'+str(split_nr)+'.py')
		split_nr+=1

		# # Average CV score on the training set was: 0.6931966449207828
		# exported_pipeline = make_pipeline(
		#     StackingEstimator(estimator=LinearSVC(C=10.0,max_iter=120000, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)),
		#     Normalizer(norm="max"),
		#     GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.2, min_samples_leaf=18, min_samples_split=20, n_estimators=100, subsample=0.6500000000000001)
		# )
		# # Fix random state for all the steps in exported pipeline
		# #set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		# exported_pipeline.fit(featuredata[train], labeldata[train])
		# results = exported_pipeline.predict(featuredata[test])
		# accuracy = accuracy_score(labeldata[test], results)
		# model_results['GDC'].append([accuracy,train,test])

		# # Average CV score on the training set was: 0.748572421636628
		# exported_pipeline = make_pipeline(
		#     StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=1.0, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=0.0)),
		#     LinearSVC(C=15.0,max_iter=120000, dual=False, loss="squared_hinge", penalty="l1", tol=1e-05)
		# )
		# # Fix random state for all the steps in exported pipeline
		# #set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		# exported_pipeline.fit(featuredata[train], labeldata[train])
		# results = exported_pipeline.predict(featuredata[test])
		# accuracy = accuracy_score(labeldata[test], results)
		# model_results['SGD'].append([accuracy,train,test])

		# # Average CV score on the training set was: 0.7373165618448637
		# exported_pipeline = make_pipeline(
		#     FastICA(tol=0.9500000000000001),
		#     LinearSVC(C=15.0,max_iter=120000, dual=False, loss="squared_hinge", penalty="l1", tol=1e-05)
		# )
		# # Fix random state for all the steps in exported pipeline
		# #set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		# exported_pipeline.fit(featuredata[train], labeldata[train])
		# results = exported_pipeline.predict(featuredata[test])
		# accuracy = accuracy_score(labeldata[test], results)
		# model_results['ICA'].append([accuracy,train,test])

		# # Average CV score on the training set was: 0.7121593291404611
		# exported_pipeline = XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=14, n_estimators=100, nthread=1, subsample=0.8500000000000001)
		# # Fix random state in exported estimator
		# #if hasattr(exported_pipeline, 'random_state'):
		# #    setattr(exported_pipeline, 'random_state', 42)

		# exported_pipeline.fit(featuredata[train], labeldata[train])
		# results = exported_pipeline.predict(featuredata[test])
		# accuracy = accuracy_score(labeldata[test], results)
		# model_results['XGB'].append([accuracy,train,test])

	#print(model_results['GDC'][0],len(model_results['GDC']))
	#print(model_results['GDC'][0][0])
	#average =[sum(x[0])/len(model_results['GDC']) for x in model_results['GDC']]
	#print(average)
	for model in model_results:
		print(model)
		average = 0
		average = [(average+value[0])/len(model_results[model]) for value in model_results[model]]
		print(sum(average))
			
			




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