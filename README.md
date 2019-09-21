# action_detection_using_neural_network
A trained neural network model to detect whether a given sentence is an actionable item or not using some pre-tagged action item sentences dataset..

#### Dataset consists of labelled sentences : 
    * True  :   825
    * False :  695

#### NaiveBayesClassifier is trained :
* Download the trained model [here](https://drive.google.com/open?id=1KVOyzOrk8SatS9_yUsN1kSMV1ZmtgriJ)
    * On 10% test - `92.7% validation accuracy` 
    * On 20% test - `88% validation accuracy`
    ) and paste it in "models" folder
    * To train the model run : `python3 main_NBC.py /path/to/data/file`
    * To predict individual sentences run : `python3 main_NBC.py "example sentence to classify"`
    
#### Convolutional Neural Network is trained :
* Download the trained model and the tokenizer [here](https://drive.google.com/open?id=1Zpt_zVloqJPMDpgjGEK0E089blgRm1wa) and paste them in "models" folder 
* To train the model run : `python3 main_CNN.py /path/to/data/file`
* To predict individual sentences run : `python3 main_CNN.py "example sentence to classify" /path
/saved_model/ /path/saved_tokenizer` 
    * The current accuracy is dropping and the model is not generalizing. 
    * Remedy is to use another network architecture ie, memory based neural networks like LSTMs.
    * The table for precision, f1 score and recall.
    
####
|     Content   | precision  |  recall | f1-score |  support
| ------------  | ---------  | ------- | -------- | ----------
|      0        |     0.00   |   0.00  |    0.00  |    23      
|      1        |     0.55   |   1.00  |    0.71  |    28   
|   accuracy    |            |         |   0.55   |    51
|   macro avg   |     0.27   |   0.50  |   0.35   |    51
| weighted avg  |     0.30   |   0.55  |   0.39   |    51

#### LSTM Recurrent Neural Network is trained :
* Download the trained model and the tokenizer [here](https://drive.google.com/open?id=1Qnn9nnRm4tDYnPo6wP9dqvsZfrhrE2tv) and paste them in "models" folder.
* To train the model run : `python3 main_LSTM.py /path/to/data/file`
* To predict individual sentences run : `python3 main_LSTM.py "example sentence to classify" /paths
/saved_model/ /path/saved_tokenizer`
    * The current accuracy `92%` on 20% validation set. 
    * Remedy is to use another network architecture ie, memory based neural networks like LSTMs.
    * The table for precision, f1 score and recall.
    
####
|     Content   | precision  |  recall | f1-score |  support
| ------------  | ---------  | ------- | -------- | ----------
|      0        |     0.92   |   0.91  |    0.92  |    144      
|      1        |     0.92   |   0.93  |    0.93  |    304   
|   accuracy    |            |         |   0.92   |    304
|   macro avg   |     0.92   |   0.92  |   0.92   |    304 
| weighted avg  |     0.92   |   0.92  |   0.92   |    304


### Need for improvement
* Including pretrained glove vector embeddings to boost accuracy.
* Increase the dataset, for deep learning large amount of data is required to boost accuracy.
* Include K-fold cross validation in LSTM training module.s