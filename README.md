# Hashtag_Recommendations_NN

Model.py is the logistic regression model we used to predict happy and FML hashtags the validation error we got with that was 92% for Bag of Words
91% for tf_idf and Bag of Words

In embeddings.ipynb, we used Twitter Glove vector for word representation which can be obtained from
https://nlp.stanford.edu/projects/glove/ trained on a neural network with 1st layer activation function = ReLU and the final activation function = sigmoid to obtain the accuracy of 90%. The neural network was then trained without pre-trained embeddings from Twitter Glove vector in the final graphs and with that we achieved an accuracy of 91% on the test data. But it performs really well on the validation data as the graphs show. 
