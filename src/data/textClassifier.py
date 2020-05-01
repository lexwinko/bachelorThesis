import pandas as pd
import numpy as np
import sys
import time
import csv

#import ktrain
#from ktrain import text as txt


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
	data = pd.read_csv(file, header=0, sep=',')
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
					'charNGrams_similarity_French',
					'wordNGrams_similarity_French',
					'charNGrams_similarity_German',
					'wordNGrams_similarity_German',
					'charNGrams_similarity_Greek',
					'wordNGrams_similarity_Greek',
					'charNGrams_similarity_Indian',
					'wordNGrams_similarity_Indian',
					'charNGrams_similarity_Russian',
					'wordNGrams_similarity_Russian',
					'charNGrams_similarity_Japanese',
					'wordNGrams_similarity_Japanese',
					'charNGrams_similarity_Turkish',
					'wordNGrams_similarity_Turkish',
					'charNGrams_similarity_English',
					'wordNGrams_similarity_English']

	le = preprocessing.LabelEncoder()
	le.fit(lang)
	print(le.classes_)

	featuredata = data[features].to_numpy()
	labeldata = le.transform(data['lang'])

	kf = KFold(n_splits=10)

	model_results = {'GDC':[], 'SGD':[], 'ICA':[], 'XGB':[]}

	for train, test in kf.split(featuredata):

		# Average CV score on the training set was: 0.6931966449207828
		exported_pipeline = make_pipeline(
		    StackingEstimator(estimator=LinearSVC(C=10.0,max_iter=120000, dual=False, loss="squared_hinge", penalty="l1", tol=0.001)),
		    Normalizer(norm="max"),
		    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.2, min_samples_leaf=18, min_samples_split=20, n_estimators=100, subsample=0.6500000000000001)
		)
		# Fix random state for all the steps in exported pipeline
		#set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		exported_pipeline.fit(featuredata[train], labeldata[train])
		results = exported_pipeline.predict(featuredata[test])
		accuracy = accuracy_score(labeldata[test], results)
		model_results['GDC'].append([accuracy,train,test])

		# Average CV score on the training set was: 0.748572421636628
		exported_pipeline = make_pipeline(
		    StackingEstimator(estimator=SGDClassifier(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=1.0, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=0.0)),
		    LinearSVC(C=15.0,max_iter=120000, dual=False, loss="squared_hinge", penalty="l1", tol=1e-05)
		)
		# Fix random state for all the steps in exported pipeline
		#set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		exported_pipeline.fit(featuredata[train], labeldata[train])
		results = exported_pipeline.predict(featuredata[test])
		accuracy = accuracy_score(labeldata[test], results)
		model_results['SGD'].append([accuracy,train,test])

		# Average CV score on the training set was: 0.7373165618448637
		exported_pipeline = make_pipeline(
		    FastICA(tol=0.9500000000000001),
		    LinearSVC(C=15.0,max_iter=120000, dual=False, loss="squared_hinge", penalty="l1", tol=1e-05)
		)
		# Fix random state for all the steps in exported pipeline
		#set_param_recursive(exported_pipeline.steps, 'random_state', 42)

		exported_pipeline.fit(featuredata[train], labeldata[train])
		results = exported_pipeline.predict(featuredata[test])
		accuracy = accuracy_score(labeldata[test], results)
		model_results['ICA'].append([accuracy,train,test])

		# Average CV score on the training set was: 0.7121593291404611
		exported_pipeline = XGBClassifier(learning_rate=0.1, max_depth=9, min_child_weight=14, n_estimators=100, nthread=1, subsample=0.8500000000000001)
		# Fix random state in exported estimator
		#if hasattr(exported_pipeline, 'random_state'):
		#    setattr(exported_pipeline, 'random_state', 42)

		exported_pipeline.fit(featuredata[train], labeldata[train])
		results = exported_pipeline.predict(featuredata[test])
		accuracy = accuracy_score(labeldata[test], results)
		model_results['XGB'].append([accuracy,train,test])

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