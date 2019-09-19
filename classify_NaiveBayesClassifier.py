from parse_data import read_data
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split

import os
import pickle

# filepath = sys.argv[1]

def train_NBC(filepath):
	new_train_test = read_data(filepath)
	x_train, x_test = train_test_split(new_train_test, test_size=0.1)
	
	cl = NaiveBayesClassifier(x_train)
	# print(cl.classify("Please create an assignment and forward it by EOD"))
	# print(cl.classify("Im not a dessert person but the warm butter cake should be illegal its so good."))
	
	print("Acheived a test accuracy of : %s " % cl.accuracy(x_test))
	
	# details of classifier train
	cl.show_informative_features()
	
	if not os.path.isdir("./models"):
		os.mkdir("./models")
	# saving the trained model
	file = open("./models/cl_NBC.obj", "wb")
	pickle.dump(cl, file)
	file.close()
	

def predict_NBC(sentence):
	classifier_f = open("./models/cl_NBC.obj", "rb")
	classifier = pickle.load(classifier_f)
	classifier_f.close()
	
	print("Predicted class of input sentence : %s\t\t is : %s" % (sentence, classifier.classify(sentence)))