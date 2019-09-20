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
    * The table for precision, f1 score and recall.
    
####
|     Content   | precision  |  recall | f1-score |  support
| ------------  | ---------  | ------- | -------- | ----------
|      0        |     1.0    |   1.00  |    1.00  |    23      
|      1        |     1.0    |   1.00  |    1.00  |    27   
|   micro avg   |     1.00   |   1.00  |   1.00   |     50
|   macro avg   |     1.00   |  1.00   |   1.00   |     50
| weighted avg  |     1.00   |   1.00  |   1.00   |     50
| samples avg   |     1.00   |   1.00  |   1.00   |     50

