import warnings
warnings.filterwarnings('ignore')

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pickle
import numpy as np
from parse_data import convert_csv_list
from keras_cnn_model import create_model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# filepath = "./old/train.csv"


def train_CNN(filepath):
	texts, labels = convert_csv_list( filepath )
	tokenizer = Tokenizer(num_words=300)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	word_index = tokenizer.word_index
	vocab_size = len(word_index)
	data = pad_sequences(sequences, maxlen=322)
	print("Length of training data {}".format(len(data)))
	print("Shape of data {}".format(data.shape))
	labels_cat = np.array(labels)

	kfold = StratifiedKFold(n_splits=30, shuffle=True, random_state=12)
	cvscores = []
	models=[]
	test_data=[]

	""" Ready to train """
	print(" data shape {}".format(data.shape))
	# print(" train shape {}".format(labels.shape))
	for train,test in kfold.split(data,labels):
		# As keras does not have support for multi filters in cnn on same output from embedding layer hence
		# proceeding with one layer of cnn with one filte
		Y = labels_cat[train]
		Y_test = labels_cat[test]
		model = create_model(vocab_size+2,100,322,(3,),256,0.3)
		model.fit(data[train],Y,epochs=2,batch_size=128)
		scores = model.evaluate(data[test],Y_test,verbose=1)
		print("{} {}".format(model.metrics,scores))
		cvscores.append(scores[1])
		models.append(model)
		test_data.append(test)
		
	max_index=np.array(cvscores).argmax()
	model = models[max_index]
	t_data = test_data[max_index]
	predicted = model.predict(data[t_data])
	print(np.round(predicted))
	print(labels_cat[t_data])
	print(classification_report(labels_cat[t_data],np.round(predicted)))
	# print(confusion_matrix(np.argmax(labels_cat[t_data]),np.argmax(np.round(predicted))))
	cvscores.append(scores)
	model.save('./models/cl_CNN_demo.h5')
	pickle.dump(tokenizer,open('./models/tokenizer_demo.p','wb'))


def predict_CNN(saved_model_path, saved_tokenizer_path, filepath):
	class_category = ["False", "True"]
	model = load_model(saved_model_path)
	tokenizer = pickle.load(open(saved_tokenizer_path, "rb"))
	
	x_pred = tokenizer.texts_to_sequences(filepath)
	x_pred = pad_sequences(x_pred, maxlen=322)
	result = model.predict( x_pred )
	print( class_category[ result.argmax() ] )