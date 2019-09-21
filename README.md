# action_detection_using_neural_network
A trained neural network model to detect whether a given sentence is an actionable item or not using some pre-tagged action item sentences dataset..

* Dataset consists of labelled sentences : 
    * True  :   825
    * False :  695

* NaiveBayesClassifier is trained :
    * On 10% test - `92.7% validation accuracy` 
    * On 20% test - `88% validation accuracy`
    * download the trained model [here](https://drive.google.com/open?id=1KVOyzOrk8SatS9_yUsN1kSMV1ZmtgriJ
    ) and paste it in "models" folder
    * To train the model run : `python3 main_NBC.py /path/to/data/file`
    * To predict individual sentences run : `python3 main_NBC.py "example sentence to classify"`
    
* Convolutional Neural Network is trained :
    * Download the trained model and the tokenizer [here]() and paste them in "models" folder 
    * Since deep learning takes massive amounts of data, the current dataset consists of many outliers and
     false
     classified labels, the current accuracy will drop
     and the model will not generalize. 
    * Remedy is to feed more data to get the weightage of outliers drop to minimum.
    * The table for precision, f1 score and recall.
    
####
|     Content   | precision  |  recall | f1-score |  support
| ------------  | ---------  | ------- | -------- | ----------
|      0        |     0.51   |   0.60  |    0.60  |    23      
|      1        |     0.55   |   1.00  |    0.71  |    28   
|   accuracy    |            |         |   0.55   |     50
|   macro avg   |     0.27   |  0.50   |   0.35   |     50
| weighted avg  |     0.30   |   0.55  |   0.39   |     50


### Need for improvement
    * Including pretrained glove vector
    * Increase the dataset, for deep learning large amount of data is required. Low recall, f1-score means
     model is not generalizing