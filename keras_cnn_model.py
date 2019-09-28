from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.core import Flatten
from keras.models import Sequential


def create_model( vocab_size, embedding_size, max_sentence_length, filter_sizes, num_filters, dropout ) :
	model = Sequential()
	model.add( Embedding( vocab_size, embedding_size, input_length=max_sentence_length ) )
	# for filter_size in filter_sizes:
	model.add( Conv1D( num_filters, 3, activation='relu' ) )
	model.add( MaxPooling1D( pool_size=(max_sentence_length - 3 + 1,), strides=1 ) )
	model.add( Dropout( dropout ) )
	model.add( Flatten() )
	model.add( Dense( 1, activation='relu' ) )
	model.add( Activation( 'sigmoid' ) )
	# model.compile(loss=keras.losses.categorical_crossentropy,optimzer=keras.optimizers.SGD(),
	# metrics=['accuracy'])
	model.compile( loss='binary_crossentropy', optimizer='adam', metrics=[ 'accuracy' ] )
	return model
