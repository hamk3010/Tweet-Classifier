import io
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression 

###########################################################################
#generate training/test data
########################################################################### 
bragging_file = io.open("cleantweets/Happy_clean.txt",'r',encoding="utf-8") 
bitching_file = io.open("cleantweets/FML_clean.txt",'r',encoding="utf-8") 


bragging_tweets = []
bitching_tweets = []

for _ in range(249000):
    bragging_tweets.append([bragging_file.readline().strip("\n"),0])
    bitching_tweets.append([bitching_file.readline().strip("\n"),1])
    
tweets = bragging_tweets+bitching_tweets
np.random.seed(42)
np.random.shuffle(tweets)


tweets_train = np.array(tweets[:348600])
tweets_test = np.array(tweets[348600:498000])

X_train = tweets_train[:,0]
Y_train = tweets_train[:,1]
X_test = tweets_test[:,0]
Y_test = tweets_test[:,1]

#########################################################################
#bag of words
#########################################################################

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

#########################################################################
#logistic regression
#########################################################################

LR = LogisticRegression().fit(X_train_counts, Y_train)
predicted_LR = LR.predict(X_test_counts)
print('Accuracy with logistic regression (BOW):....',(np.round(np.mean(predicted_LR==Y_test),2)))
#uncomment to see results
'''
for i in range(50): 
    predicted_LR = LR.predict(X_test_counts[i])
    try:
        print(predicted_LR)
        print(X_test[i],Y_test[i])
    except:
        print("error")
'''    

LR = LogisticRegression().fit(X_train_counts, Y_train) 
predicted_LR = LR.predict(X_test_tfidf)

print('Accuracy with logistic regression (tfidf):....',(np.round(np.mean(predicted_LR==Y_test),2)))

#uncomment to see results
'''
for i in range(50): 
    predicted_LR = LR.predict(X_test_counts[i])
    try:
        print(predicted_LR)
        print(X_test[i],Y_test[i])
    except:
        print("error")
'''    
