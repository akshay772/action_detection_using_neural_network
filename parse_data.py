from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

import tensorflow as tf
import pandas as pd

# filepath = sys.argv[ 1 ]
# names = ["label", "sent"]


def read_data( filepath ) :
	# no of columns founds in csv to read through
	my_cols = [ ]
	for i in range( 1, 12 ) :
		my_cols.append( str( i ) )
	df = pd.read_csv( filepath, names=my_cols, skiprows=1 )
	df = df.fillna( "" )
	df[ "new" ] = df.astype( str ).values.sum( axis=1 )
	df[ "sent_label" ] = df[ "new" ].str.split( "|", 1 )
	new_df = pd.DataFrame()
	new_df[ "sent" ], new_df[ "label" ] = df[ "new" ].str.split( "|", 1 ).str
	new_df = new_df.reset_index()
	del new_df[ "index" ]
	new_df.label = new_df.label.str.replace( " ", "" )
	
	return new_df


def convert_csv_list( filepath ) :
	texts = [ ]
	labels = [ ]
	new_df = read_data( filepath )
	new_df.label = new_df.label.str.replace( "True", "1" )
	new_df.label = new_df.label.str.replace( "False", "0" )
	texts = new_df.sent.values.tolist()
	labels = new_df.label.values.tolist()
	labels = list( map( int, labels ) )
	
	return texts, labels


def to_one_hot(y, n_class):
	return np.eye(n_class)[y.astype(int)]


def read_df_to_series(filepath, sample_ratio=1, no_class=2,  one_hot=True):
	csv_file = read_data( filepath )
	csv_file.label = csv_file.label.str.replace( "True", "1" )
	csv_file.label = csv_file.label.str.replace( "False", "0" )
	shuffle_csv = csv_file.sample( frac=sample_ratio )
	input_data = pd.DataFrame(shuffle_csv.sent)
	target_data = pd.DataFrame(shuffle_csv.label)
	X_train, X_test, y_train, y_test = train_test_split( input_data, target_data, test_size=0.2 )
	X_train = pd.Series(X_train["sent"])
	X_test = pd.Series( X_test[ "sent" ] )
	y_train = pd.Series(y_train["label"])
	y_test = pd.Series(y_test["label"])
	
	if one_hot :
		y_train = to_one_hot( y_train, no_class )
		y_test = to_one_hot( y_test, no_class)
	
	return X_train, y_train, X_test, y_test


def data_preprocessing(x_train, x_test, max_len, max_words=30000):
	tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
	tokenizer.fit_on_texts(x_train)
	# for converting raw text to input sequences to cnn
	train_index = tokenizer.texts_to_sequences(x_train)
	test_index = tokenizer.texts_to_sequences(x_test)
	# introduce padding so every sequence will be equal length
	train_padded = pad_sequences(train_index, maxlen=max_len, padding='post', truncating='post')
	test_padded = pad_sequences(test_index, maxlen=max_len, padding='post', truncating='post')
	
	return train_padded, test_padded, max_words + 2


def split_dataset(x_test, y_test, dev_ratio):
	# split test dataset to test and dev set with ratio
	test_size = len(x_test)
	print(test_size)
	dev_size = (int)(test_size * dev_ratio)
	print(dev_size)
	x_dev = x_test[:dev_size]
	x_test = x_test[dev_size:]
	y_dev = y_test[:dev_size]
	y_test = y_test[dev_size:]
	
	return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


# def fill_feed_dict(data_X, data_Y, batch_size):
# 	# Generator to yield batches
# 	# Shuffle data first
# 	shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
# 	for idx in range(data_X.shape[0] // batch_size):
# 		x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
# 		y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
# 		yield x_batch, y_batch