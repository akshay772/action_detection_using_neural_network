from sklearn.model_selection import train_test_split

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