from classify_NaiveBayesClassifier import train_NBC, predict_NBC

import os
import sys

filepath = sys.argv[1]
if not os.path.isfile(filepath):
	if not os.path.isfile("./models/cl_NBC.obj"):
		print("There is no model trained... run training module first.")
		sys.exit()
	# do sentence predicting
	predict_NBC(filepath)

else:
	# do training NBC first
	train_NBC(filepath)