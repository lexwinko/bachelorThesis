import pandas as pd
import numpy as np
import sys
import time
import csv

#import ktrain
#from ktrain import text as txt


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import KFold
from sklearn import preprocessing

from tpot import TPOTClassifier


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

	print(labeldata)
			


	###################################################
	#			
	#			KTRAIN
	#
	# data_ktrain = pd.DataFrame(data['originalSentence']).join(classes)

	# print(data_ktrain.head())

	# (x_train, y_train), (x_test, y_test), preproc = txt.texts_from_df(data_ktrain, 
	# 															   'originalSentence', # name of column containing review text
	# 															   label_columns=lang,
	# 															   maxlen=40, 
	# 															   max_features=100000,
	# 															   preprocess_mode='bert',
	# 															   val_pct=0.1,
	# 															   ngram_range=2)

	# print(preproc)

	# model = txt.text_classifier('bert', (x_train, y_train) , preproc=preproc)
	# learner = ktrain.get_learner(model, 
	# 						 train_data=(x_train, y_train), 
	# 						 val_data=(x_test, y_test), 
	# 						 batch_size=32)

	# learner.lr_find(show_plot=True)
	# learner.autofit(5e-3, 1)

	# learner.view_top_losses(n=1, preproc=preproc)


	###################################################
	#		
	#			Other models
	#
	#data_train_X, data_test_X, data_train_y, data_test_y = train_test_split(featuredata, data['lang'], test_size=0.3, random_state=int(time.time()))

	kf = KFold(n_splits=10)
	tpot = TPOTClassifier(n_jobs=-1, generations=5, population_size=50, verbosity=2, random_state=42)


	splitnr = 0
	for train, test in kf.split(featuredata):
		print(len(train), len(test))
		tpot.fit(featuredata[train], labeldata[train])
		print(tpot.score(featuredata[test], labeldata[test]))
		tpot.export('tpot_digits_pipeline_'+str(splitnr)+'.py')
		splitnr += 1

		# if method == 'GNB':
		# 	gnb = GaussianNB()
		# 	model = gnb.fit(featuredata[train], labeldata[train])
		# 	print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(gnb.score(featuredata[train], labeldata[train])))
		# 	print('Accuracy of GaussianNB regression classifier on test set: {:.2f}'.format(gnb.score(featuredata[test], labeldata[test])))

		# elif method == 'LR':
		# 	LR = LogisticRegression(random_state=int(time.time()), solver='lbfgs', multi_class='multinomial').fit(featuredata[train], labeldata[train])
		# 	print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(LR.score(featuredata[train], labeldata[train])))
		# 	print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(LR.score(featuredata[test], labeldata[test])))

		# elif method == 'SVM':
		# 	SVM = svm.SVC(decision_function_shape="ovo", gamma='auto').fit(featuredata[train],labeldata[train])
		# 	print('Accuracy of SVM classifier on training set: {:.2f}'.format(SVM.score(featuredata[train], labeldata[train])))
		# 	print('Accuracy of SVM classifier on test set: {:.2f}'.format(SVM.score(featuredata[test], labeldata[test])))

		# elif method == 'RF':
		# 	RF = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=int(time.time())).fit(featuredata[train], labeldata[train])
		# 	print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(RF.score(featuredata[train], labeldata[train])))
		# 	print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(featuredata[test], labeldata[test])))

		# elif method == 'NN':
		# 	NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=int(time.time())).fit(featuredata[train], labeldata[train])
		# 	print('Accuracy of Neural network classifier on training set: {:.2f}'.format(NN.score(featuredata[train], labeldata[train])))
		# 	print('Accuracy of Neural network classifier on test set: {:.2f}'.format(NN.score(featuredata[test], labeldata[test])))
	





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