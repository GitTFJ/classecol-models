# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#load packages
import numpy as np
import os
import pandas as pd
import re
import nltk
nltk.download('wordnet')
from bs4 import BeautifulSoup
import string
import joblib
import pickle
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, metrics
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import keras
from keras.preprocessing import text, sequence
from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.backend as K
import tensorflow as tf
import time

#set seed
np.random.seed(10)
os.chdir(r.model_directory)

#clean data
replace_with_space = re.compile('[/(){}\[\]\|@,;]')
bad_symbols = re.compile('[^0-9a-z #+_]')
stopword = ['i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'between',
 'to',
 'from',
 'then',
 'here',
 'there',
 'when',
 'where',
 'so',
 'than']

ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()

def clean(text):
    text = text.lower() # lowercase text
    text = re.sub('[0-9]+', '', text)
    text = BeautifulSoup(text, "html.parser").text # HTML decoding
    text = replace_with_space.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = bad_symbols.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text

def contract(text):
    text = [wn.lemmatize(word) for word in text]
    text = [ps.stem(word) for word in text]
    text  = "".join([char for char in text if char not in string.punctuation])
    text = ' '.join(word for word in text.split() if word not in stopword) # delete stopwors from text
    return text

r.data['text'] = r.data['text'].astype(str)
r.data['text'] = r.data['text'].apply(clean)
r.data['text'] = r.data['text'].apply(lambda x: contract(x))
r.data['text'].apply(lambda x: len(x.split(' '))).sum()

#load dat
nb = joblib.load('bio_per_org_rapid_nb.sav')
svm = joblib.load('bio_per_org_rapid_svm.sav')
rf = joblib.load('bio_per_org_rapid_rf.sav')
lr = joblib.load('bio_per_org_rapid_lr.sav')
nn = joblib.load('bio_per_org_rapid_nn.sav')
dnn = keras.models.load_model("bio_per_org_rapid_dnn.h5")

with open('bio_per_org_rapid_tokenize', 'rb') as handle:
    tokenize = pickle.load(handle)
tok = tokenize.transform(r.data['text']).toarray()
with open('bio_per_org_rapid_norm', 'rb') as handle:
    norm = pickle.load(handle)
tok = norm.transform(tok)

pred_nb = nb.predict_proba(r.data['text'])
pred_svm = svm.decision_function(r.data['text'])[:, None]
pred_rf = rf.predict_proba(r.data['text'])
pred_lr = lr.predict_proba(r.data['text'])
pred_nn = nn.predict_proba(r.data['text'])
pred_dnn = dnn.predict(tok)

#join probabiliy predictions
comb = np.concatenate([
    pred_nb, 
    pred_svm, 
    pred_rf, 
    pred_lr, 
    pred_nn, 
    pred_dnn
    ], axis=1)
    
ensemble = joblib.load('bio_per_org_rapid_ensemble.sav')
pred_comb = ensemble.predict(comb)
