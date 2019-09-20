from keras_train_CNN import train_CNN, predict_CNN

import os
import sys

filepath = sys.argv[1]

if not os.path.isfile(filepath):
	saved_model_path = sys.argv[2]
	print("Model is saved in : %s", saved_model_path)
	saved_tokenizer_path = sys.argv[3]
	print( "Tokenizer is saved in : %s", saved_tokenizer_path )
	if not os.path.isfile("./models/cl_CNN.h5"):
		print("There is no model trained... run training module first.")
		sys.exit()
	if not os.path.isfile("./models/tokenizer.p"):
		print("Tokenizer is absent from the trained model. Paste it to models folder")
		sys.exit()
	# do sentence predicting
	predict_CNN(saved_model_path, saved_tokenizer_path, filepath)
	
else:
	# do training CNN first
	train_CNN(filepath)
