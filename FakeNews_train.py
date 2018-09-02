# Importing necessary packages
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
from textblob import TextBlob
import gensim
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import pickle
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.utils import np_utils

# Set the directory
os.chdir('C:/Users/mansiarora/Documents/CSE 6242/Project/Scripts')
import Fakenews_featureextraction as fe

# Reading the dataset
faketrain=pd.read_csv('dataset.csv', encoding='latin-1')

# Cleaning the dataset
datatrain = fe.clean_text_add_features(faketrain)

# Converting words to vectors using Google's word2vec file
NLPmodel = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
datatrain=fe.word_vectors(datatrain)

# Final structuring of the file
finaldata = datatrain[['fake', 'specialchar','wordcount','avewordlength', 'firstpersonwordcount', 'uniquewords', 'capitalizedwords', 'vector']]
finaldata2=pd.concat([finaldata['vector'].apply(pd.Series), finaldata], axis = 1)
finaldata2.drop('vector', axis=1, inplace=True)

finaldata2.dropna(inplace=True)

# Creating train and test files
finaldata_predictors = finaldata2.drop('fake',axis=1)
finaldata_response = finaldata2.as_matrix(['fake'])
finaldata_predictor = finaldata_predictors.as_matrix()
finaldatatrain_predictors, finaldatatest_predictors, finaldatatrain_response,  finaldatatest_response = train_test_split(finaldata_predictor, finaldata_response, test_size=0.2, random_state=40)

# Baseline Model using SVM
#svmmodel = svm.SVC()
#svmmodel.fit(finaldatatrain_predictors, finaldatatrain_response)
#svm_test_output = svmmodel.predict(finaldatatest_predictors)
#accuracy_score(finaldatatest_response, svm_test_output)

# XGBOOST Model
#xgbmodel = xgb.XGBClassifier(nthread=10, silent=False, max_depth=7, learning_rate=0.2)
#xgbmodel.fit(finaldatatrain_predictorssvd, finaldatatrain_response)
#xgb_test_output = xgbmodel.predict(finaldatatest_predictorssvd)
#accuracy_score(finaldatatest_response, xgb_test_output)
#pickle.dump(xgbmodel, open("fakenews_model.dat", "wb"))

# Building the Deep Learning Model

# Scaling the data before any neural net:
scl = preprocessing.StandardScaler()
xtrain_scl = scl.fit_transform(finaldatatrain_predictors)
xvalid_scl = scl.transform(finaldatatest_predictors)

# Binarize the labels for the neural net
ytrain_enc = np_utils.to_categorical(finaldatatrain_response)
yvalid_enc = np_utils.to_categorical(finaldatatest_response)

# Creating a simple 1 layer sequential neural net

model = Sequential()

model.add(Dense(306, input_dim=306, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(2))
model.add(Activation('softmax'))

# Compiling the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Storing callbacks for plotting the graph 
outputFolder = './output-mnist'
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)
filepath=outputFolder+"/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, 
                             save_best_only=False, save_weights_only=True, 
                             mode='auto', period=10)
callbacks_list = [checkpoint]

# Fitting the model
model_info = model.fit(xtrain_glove_scl, y=ytrain_enc, batch_size=64, 
          callbacks=callbacks_list, epochs=60, verbose=1, validation_split=0.3)

# Plotting the plots for accuracy
fig, axs = plt.subplots(1,2,figsize=(15,5))
axs[0].plot(range(1,len(model_info.history['acc'])+1),model_info.history['acc'])
axs[0].plot(range(1,len(model_info.history['val_acc'])+1),model_info.history['val_acc'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_xticks(np.arange(1,len(model_info.history['acc'])+1),len(model_info.history['acc'])/10)
axs[0].legend(['Training', 'Validation'], loc='best')
axs[1].plot(range(1,len(model_info.history['loss'])+1),model_info.history['loss'])
axs[1].plot(range(1,len(model_info.history['val_loss'])+1),model_info.history['val_loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_xticks(np.arange(1,len(model_info.history['loss'])+1),len(model_info.history['loss'])/10)
axs[1].legend(['Training', 'Validation'], loc='best')
plt.show()

# Predictions on the validation set
model.predict(xvalid_glove_scl)
prediction = np.argmax(model.predict(xvalid_glove_scl),1)

# Accuracy score
accuracy_score(prediction, finaldatatest_response)

# Saving the final model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")


