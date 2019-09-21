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


def to_one_hot(y, n_class):
	return np.eye(n_class)[y.astype(int)]


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