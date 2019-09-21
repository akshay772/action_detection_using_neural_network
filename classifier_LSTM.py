import numpy
import pickle
import tensorflow as tf

from parse_data import convert_csv_list

from numpy import array
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# filepath = "./old/train.csv"


def train_LSTM(filepath):
	texts, labels = convert_csv_list(filepath)
	X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
	y_train = array(y_train)
	y_test = array(y_test)
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=500)
	tokenizer.fit_on_texts(X_train)
	X_train = tokenizer.texts_to_sequences(X_train, )
	X_test = tokenizer.texts_to_sequences(X_test)
	X_train = array(X_train)
	X_test = array(X_test)
	numpy.random.seed(7)
	top_words = 500
	# truncate and pad input sequences
	max_review_length = 322
	X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
	X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
	# create the model
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(Dropout(0.2))
	model.add(LSTM(100))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(X_train, y_train, epochs=30, batch_size=64)
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	y_pred = model.predict(X_test)
	y_pred = (y_pred>0.5)
	print(classification_report(y_test, y_pred))
	print("Accuracy: %.2f%%" % (scores[1]*100))
	
	model.save( './models/cl_LSTM.h5' )
	pickle.dump( tokenizer, open( './models/tokenizer_LSTM.p', 'wb' ) )


def predict_LSTM( saved_model_path, saved_tokenizer_path, filepath ) :
	class_category = [ "True", "False" ]
	model = load_model( saved_model_path )
	tokenizer = pickle.load( open( saved_tokenizer_path, "rb" ) )
	
	x_pred = tokenizer.texts_to_sequences( filepath )
	x_pred = sequence.pad_sequences( x_pred, maxlen=322 )
	result = model.predict( x_pred )
	print( class_category[ result.argmax() ] )

