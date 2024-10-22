# -*- coding: utf-8 -*-
"""
Created on mon 22/10/2024  23:25:15 2024

@author: Abhishek Singh
"""

import pandas as pd
# dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
messages = pd.read_csv(r"sms+spam+collection/SMSSpamCollection",
                       sep='\t',names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model (BOW)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model_BOW = MultinomialNB()
model_BOW.fit(X_train, y_train)
y_pred=model_BOW.predict(X_test)


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accuracy with the use of  BOW:", accuracy_score(y_test, y_pred))
print("confusion_matrix :\n", confusion_matrix(y_test, y_pred))
print("classification_report :", classification_report(y_test, y_pred))


#TF_IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfvectorizer = TfidfVectorizer(max_features=2500)
X = tfvectorizer.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
model_TFIDF = MultinomialNB().fit(X_train, y_train)

y_pred=model_TFIDF.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print("accuracy with the use of  TF-IDF:", accuracy_score(y_test, y_pred))
print("confusion_matrix :\n", confusion_matrix(y_test, y_pred))
print("classification_report :", classification_report(y_test, y_pred))


# The accuracy of BOW is more than TF-IDF

import pickle
with open(r'Model/modelforPrediction.pkl', 'wb') as file:
    pickle.dump(model_BOW, file)

with open(r'Model/vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)
