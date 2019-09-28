# action_detection_using_neural_network
A trained neural network model to detect whether a given sentence is an actionable item or not using some pre-tagged action item sentences dataset..

##### Requirement :
* To install dependencies run `pip install -r requirements.txt` 

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
    * The current accuracy `95%` on 20% validation set.
    * The table for precision, f1 score and recall.

####
|     Content   | precision  |  recall | f1-score |  support
| ------------  | ---------  | ------- | -------- | ----------
|      0        |     0.98   |   0.90  |   0.94   |    69
|      1        |     0.92   |   0.99  |   0.95   |    82
|   accuracy    |            |         |   0.95   |    151
|   macro avg   |     0.95   |   0.94  |   0.95   |    151
| weighted avg  |     0.95   |   0.95  |   0.95   |    151

#### LSTM Recurrent Neural Network is trained :
* Download the trained model and the tokenizer [here](https://drive.google.com/open?id=1Qnn9nnRm4tDYnPo6wP9dqvsZfrhrE2tv) and paste them in "models" folder.
* To train the model run : `python3 main_LSTM.py /path/to/data/file`
* To predict individual sentences run : `python3 main_LSTM.py "example sentence to classify" /paths
/saved_model/ /path/saved_tokenizer`
    * The current accuracy `92%` on 20% validation set.
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
* Include K-fold cross validation in LSTM training module.
