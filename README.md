# action_detection_using_neural_network
A trained neural network model to detect whether a given sentence is an actionable item or not using some pre-tagged action item sentences dataset..

* Dataset consists of labelled sentences : 
    * True  :   825
    * False :  695

* NaiveBayesClassifier is trained :
    * On 10% test - `92.7% validation accuracy` 
    * On 20% test - `86% validation accuracy`
    * download the trained model [here](https://drive.google.com/open?id=1KVOyzOrk8SatS9_yUsN1kSMV1ZmtgriJ
    ) and paste it in "models" folder
    * To train the model run : `python3 main_NBC.py /path/to/data/file`
    * To predict individual sentences run : `python3 main_NBC.py "example sentence to classify"`
    
