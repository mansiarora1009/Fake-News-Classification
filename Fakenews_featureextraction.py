import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import os
#from textblob import TextBlob
import gensim
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
#import xgboost as xgb
import pickle


def clean_text_add_features (dataframe):
    
    text=dataframe.copy()   
    

    text['text'] =  text['text'].apply(lambda x: re.sub('[\\t\\r\\n]','',str(x)))
    text['text']= text['text'].apply(lambda x: x.encode("ascii", "replace"))
    text['specialchar']=text['text'].apply(lambda x: len([l for l in str(x) if not (str(l).isalnum() or str(l).isspace())])/len([l for l in str(x)]))
    
    #remove ? that are generated after the ascii conversion of non unicode characters
    text['text']=  text['text'].apply(lambda x: re.sub('[?]',' ',str(x)))
    
    #remove url
    text['text']=  text['text'].apply(lambda x: re.sub('[a-zA-Z]+[.]{1}[a-zA-Z0-9]+[.]*[a-zA-Z0-9//]*',' ',str(x)))
    text['text']=  text['text'].apply(lambda x: re.sub('[^a-zA-Z ]','',str(x)))
    
    #feature extraction
    text['words']= text['text'].apply(lambda x: str(x[1:]).split())
    text['text']= text['words'].apply(lambda x: ' '.join(x))
    text['wordcount']=text['words'].apply(lambda x: len(x))
    
    #filter to include only those rows where the wordcount is >0
    text=text[(text.wordcount>0)]
    
    
    text['avewordlength']=text['words'].apply(lambda x: sum([len(l) for l in x])/len([len(l) for l in x]))
    text['firstpersonwordcount']=text['words'].apply(lambda x: len([l for l in x if (str(l).lower()=='i' or str(l).lower()=='we' or str(l).lower()=='my')])/len([w for w in x]))
    text['uniquewords']=text['words'].apply(lambda x: len(set(x))/len(x))
    text['capitalizedwords']= text['words'].apply(lambda x: len([w for w in x if not w.islower()])/len([w for w in x]))
    text['text']=text['text'].apply(lambda x: str(x).lower())
    
    return text


def word_vectors(df):
    NLPmodel = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    dataframe=df.copy()
    
    dataframe['vectorlist']= dataframe['words'].apply(lambda x: [NLPmodel[w] for w in x if w in NLPmodel.vocab])
    dataframe['vector']= dataframe['vectorlist'].apply(lambda x: np.nanmean(x, axis=0))
    
    return dataframe
