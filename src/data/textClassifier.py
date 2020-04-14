import pandas as pd
import numpy as np
import sys
import time
import csv
#import ktrain
#from ktrain import text as txt
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#from tensorflow.keras import layers

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def classifyData(file, method):
	data = pd.read_csv(file, header=0, sep=',')
	data.sample(frac=1., random_state=int(time.time()))
	classes = pd.get_dummies(pd.Series(list(data['lang'])))
	lang = ['French', 'German', 'Greek', 'English', 'Indian', 'Japanese', 'Russian', 'Turkish']
	sentence = ['correctedSentence', 'originalSentence']
	ngrams = ['charNGrams','wordNGrams']
	features = ['elongated','caps','sentenceLength','sentenceWordLength','spellDelta','#','@','E',',','~','U','A','D','!','N','P','O','R','&','L','Z','^','V','$','G','T','X','S','Y','M']


	data_train_X, data_test_X, data_train_y, data_test_y = train_test_split(data[features], data['lang'], test_size=0.2, random_state=int(time.time()))

	print(len(data_train_X), len(data_train_y), len(data_test_X), len(data_test_y))


	#print(text_data)
	#print(model_data)
	#used_features_X = ['len', '#', '@', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V','elong']
	#used_features_output = ['len', '#', '@', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V','elong','lang']

	#if(predfile is not ""):
	#	predict = pd.read_csv(predfile, sep=',')[used_features_X]
	#	predict_y = pd.DataFrame(-1, index=np.arange(predict.shape[0]), columns=['val'])

	#print(data['lang'])
	#le = LabelEncoder()
	#le.fit(data['lang'])
	#print(list(le.classes_))
	#data['lang'] = le.transform(data['lang'])

	#print(data['lang'])
	#le.fit(predict['lang'])

	#print(predict)


	#le = LabelEncoder()
	#text_data['lang'] = le.fit_transform(text_data['lang'])

	#ce_oneFit = ce.OneHotEncoder(cols= ['lang'])
	#X_train = ce_oneFit.fit_transform(model_data[used_features_X], text_data)
	#text_data = text_data[used_features_y]
	#for header in list(X_train):
	#	if(header in text_data): continue
	#	text_data[header] = 0

	#print(text_data)
	#print(encoded_data)
	
	#encoded_data = pd.get_dummies(model_data[used_features])
	#encoded_labels = pd.get_dummies(model_data['lang'])

	#print(encoded_data)
	#print(encoded_labels)
	#y_train = pd.DataFrame.idxmax(y_train, axis=1)
	#print(y_train)
	#y_train, y_test = train_test_split(text_data, test_size=0.2, random_state=int(time.time()))

	#print(X_train.shape,y_train.shape)

	#print(data_train)

	#data_train_X, data_test_X, data_train_y, data_test_y = train_test_split(data[used_features_X], data['lang'], test_size=0.2, random_state=int(time.time()))

	#print(len(data_train_X), len(data_train_y), len(data_test_X), len(data_test_y))

	#data_train_X = data_train[used_features_X]
	#data_test_X = data_test[used_features_X]

	#data_train_y = data_train_X['lang']
	#data_test_y = data_test_X['lang']

	if method == 'GNB':
		gnb = GaussianNB()
		model = gnb.fit(data_train_X, data_train_y)
		print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(gnb.score(data_train_X, data_train_y)))
		print('Accuracy of GaussianNB regression classifier on test set: {:.2f}'.format(gnb.score(data_test_X, data_test_y)))

	if method == 'LR':
		LR = LogisticRegression(random_state=int(time.time()), solver='lbfgs', multi_class='multinomial').fit(data_train_X, data_train_y)
		print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(LR.score(data_train_X, data_train_y)))
		print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(LR.score(data_test_X, data_test_y)))

	if method == 'SVM':
		SVM = svm.SVC(decision_function_shape="ovo", gamma='auto').fit(data_train_X, data_train_y)
		print('Accuracy of SVM classifier on training set: {:.2f}'.format(SVM.score(data_train_X, data_train_y)))
		print('Accuracy of SVM classifier on test set: {:.2f}'.format(SVM.score(data_test_X, data_test_y)))

	if method == 'RF':
		RF = RandomForestClassifier(n_estimators=1000, max_depth=15, random_state=int(time.time())).fit(data_train_X, data_train_y)
		print('Accuracy of Random Forest classifier on training set: {:.2f}'.format(RF.score(data_train_X, data_train_y)))
		print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(data_test_X, data_test_y)))

	if method == 'NN':
		NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=int(time.time())).fit(data_train_X, data_train_y)
		print('Accuracy of Neural network classifier on training set: {:.2f}'.format(NN.score(data_train_X, data_train_y)))
		print('Accuracy of Neural network classifier on test set: {:.2f}'.format(NN.score(data_test_X, data_test_y)))

	# if method == 'LSTM':
	# 	epochs = 20
	# 	n_classes = 1
	# 	n_units = 200
	# 	n_features = len(features)
	# 	batch_size = 100

	# 	xplaceholder = tf.placeholder('float',[None, n_features])
	# 	yplaceholder = tf.placeholder('float')

	# 	def recurrent_neural_network_model():
	# 		layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}

	# 		x = tf.split(xplaceholder, n_features, 1)
	# 		print(x)

	# 		lstm_cell = rnn.BasicLSTMCell(n_units)
			
	# 		outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
		   
	# 		output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

	# 		return output

	# 	def train_neural_network():
	# 		logit = recurrent_neural_network_model()
	# 		logit = tf.reshape(logit, [-1])

	# 		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yplaceholder))
	# 		optimizer = tf.train.AdamOptimizer().minimize(cost)
	# 		optimizer = tf.compat.v1.train.AdamOptimizer().minimize(cost)

	# 		with tf.Session() as sess:
	# 			with tf.compat.v1.Session() as sess:

	# 				tf.compat.v1.global_variables_initializer().run()
	# 				tf.compat.v1.local_variables_initializer().run()
	# 				sess.run(tf.initialize_all_variables())
					
	# 				for epoch in range(epochs):
	# 					epoch_loss = 0

	# 					i = 0
	# 					for i in range(int(len(data_train_X) / batch_size)):

	# 						start = i
	# 						end = i + batch_size

	# 						batch_x = np.array(data_train_X[start:end])
	# 						batch_y = np.array(data_train_y[start:end])
	# 						print(predict_y)
							
	# 						_, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
	# 						epoch_loss += c
	# 						i += batch_size
	# 						epoch_x, epoch_y = mnist.train.next_batch(batch_size)
	# 						_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
	# 						epoch_loss += c

	# 					print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

	# 			preds = tf.round(tf.nn.sigmoid(logit)).eval({xplaceholder: np.array(data_train_X), yplaceholder: np.array(data_train_y)})
	# 			f1 = f1_score(np.array(data_train_y), preds, average='macro')
	# 			accuracy=accuracy_score(np.array(data_train_y), preds)
	# 			recall = recall_score(y_true=np.array(data_train_y), y_pred= preds)
	# 			precision = precision_score(y_true=np.array(data_train_y), y_pred=preds)
	# 			print("F1 Score:", f1)
	# 			print("Accuracy Score:",accuracy)
	# 			print("Recall:", recall)
	# 			print("Precision:", precision)

	# 			preds = tf.round(tf.nn.sigmoid(logit)).eval({xplaceholder: np.array(data_test_X), yplaceholder: np.array(data_test_y)})
	# 			f1 = f1_score(np.array(data_test_y), preds, average='macro')
	# 			accuracy=accuracy_score(np.array(data_test_y), preds)
	# 			recall = recall_score(y_true=np.array(data_test_y), y_pred= preds)
	# 			precision = precision_score(y_true=np.array(data_test_y), y_pred=preds)
	# 			print("F1 Score:", f1)
	# 			print("Accuracy Score:",accuracy)
	# 			print("Recall:", recall)
	# 			print("Precision:", precision)
	# 		return preds

	

	#row = 0
	#for x in preds.values:
	#	text_data.at[row, x] = 1
	#	row += 1

	#fields = [ 'len', '#', '@', 'E', ",", '~', 'U', 'A', 'D', '!', 'N', 'P', 'O', 'R', '&', 'L', 'Z', '^', 'V', '$', 'G', 'T', 'X', 'S', 'Y', 'M' ,'elong']
	#for x in list(X_train)[20:] :
	#	fields.append(x)

	#print(preds)

	#preds = pd.DataFrame(le.inverse_transform(preds), columns=['lang'])
	#print(preds)
	#print(preds['lang'].value_counts())

	#preds.to_csv("prediction"+method+".csv", index=None, header=True)
	#return text_data

if __name__ == "__main__":
	data = sys.argv[1]
	method = sys.argv[2]
	print('Classifying '+ data + ' with ' +method)
	classifyData(data, method)
	print('Done')
	#print(result)