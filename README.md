# Hashtag_Classification

For our data mining process we used [Jefferson-Henrique's/GetOldTweets-python](https://github.com/Jefferson-Henrique/GetOldTweets-python)

Model.py is the logistic regression model we used to predict happy and FML hashtags the validation error we got with that was 92% for Bag of Words
91% for tf_idf and Bag of Words

In Embedding.ipynb, we used [Twitter Glove vector for word representation](https://nlp.stanford.edu/projects/glove/ ) trained on a neural network with 1st layer activation function = ReLU and the final activation function = sigmoid to obtain the accuracy of 90%. The neural network was then trained without pre-trained embeddings from Twitter Glove vector in the final graphs and with that we achieved an accuracy of 91% on the test data. But it performs really well on the validation data as the graphs show.

Optimized_embedding.ipynb is the optimized version of the Embedding.ipynb file here, we're training without the pre-trained weights, we use a duel model approach where we train one model on bitching and one on bragging. Then both are tested on bitching, bragging, and neither tweets. Then if bitching outputs a score higher than 50%, we label the tweet as bitching. If bragging outputs a score higher than 50% then the tweet is marked as bragging. If they both have a score higher than 50% then the tweet is marked as both (bitching and bragging). If they both have a score higher than 50% then the tweet is marked as neither. Additionally, we optimized the model where it now takes ReLU after the first input, and softmax instead of sigmoid for the final one.

SimpleRNN.ipynb uses one simple recurrent network layer and final layer with the sigmoid function to obtain an classifying accuracy of 92% on test data. (takes a while to train)

We're still working on updating the Github and bringing it up to speed with our findings until then you can view our report at: [Report Link](https://docs.google.com/document/d/1xffgs0fqMzrzvINqbOXSkrlccr0DsVIU2Wh7T6cvDBg/edit?usp=sharing)
