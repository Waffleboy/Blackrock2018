from _pickle import dump, load

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from keras.models import Sequential, load_model
from keras.layers import LSTM, Conv1D, Dense, Flatten, MaxPooling1D, Dropout, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

#loading pre-prepared requirements
vect_ml = load(open('./scripts/vectorizer_ML.pkl', 'rb'))
tokenizer_dl = load(open('./scripts/tokenizer.pkl', 'rb'))
#embedded_seq = load(open('embedded_seq.pkl', 'rb'))

#text preprocessor class
class textpreprocessor():
    def __init__(self):
        pass
    
    #common method
    def preprocess(self, x):
        x = pd.Series(x)
        x = x.apply(str.lower)
        #replaces any urls with URL
        x = x.str.replace(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ')
        #removes user mentions
        x = x.str.replace(r"@\S+", "")
        #removes hashtags
        x = x.str.replace(r"#\S+", "")
        #removes rt
        x = x.str.replace(r'\brt\b', '')
        #removes multiple white spaces 
        x = x.str.replace(r'\s+', " ")
        #strips both ends of white space
        x = x.apply(str.strip)

        return x
    
    #ML methods
    def tokenize_ml(self, x, vect):
        vectorized_x = vect.transform(x)
    
        return vectorized_x
    
    #DL methods
    def tokenize_dl(self, x, tokenizer):
        tokenized_x = tokenizer.texts_to_sequences(x)
        
        return tokenized_x
        
    def get_num_word_dl(self, tokenizer):
        return len(tokenizer.word_counts) + 1
    
    def pad_x(self, x, max_len):
        padded_x = pad_sequences(x, maxlen = max_len, padding = 'post')
        
        return padded_x

preprocessor = textpreprocessor()

#ML models class
class ML_Models():
    def __init__(self, models, num_classes, textpreprocessor):
        self.models = models
        self.num_classes = num_classes
        self.textpreprocessor = textpreprocessor
        
    def fit(self, x, y):
        x = self._preprocess_x(x)
        models = self.models
        
        for model in models:
            model.fit(x, y)
            
    def predict(self, xtest):
        xtest = self._preprocess_x(xtest)
        models = self.models
        num_classes = self.num_classes
        test_score = np.zeros((xtest.shape[0], num_classes * len(models)))
        
        for model_num, model in enumerate(models):
            test_score[:, model_num * num_classes: model_num * num_classes + num_classes] = model.predict_proba(xtest)
        return test_score
    
    def get_models(self):
        return self.models
    
    def _preprocess_x(self, x):
        x = self.textpreprocessor.tokenize_ml(x, vect_ml)
        
        return x

#LSTM class
class LSTM_Model():
    def __init__(self, num_words, embedded_seq, max_length, num_classes, textpreprocessor):
        self.textpreprocessor = textpreprocessor
        self.max_length = max_length
        
        self.model = Sequential()
        self.model.add(Embedding(num_words, embedded_seq.shape[1], weights = [embedded_seq], input_length = max_length, trainable = False))
        self.model.add(LSTM(100, recurrent_dropout = 0.35))
        self.model.add(Dense(num_classes, activation = 'softmax'))
    
    def compile_model(self, loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']):
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        
    def fit(self, x, y):
        x = self._preprocess_x(x)
        y = self._preprocess_y(y)
        self.model.fit(x, y, epochs = 20)
        
    def predict(self, xtest):
        xtest = self._preprocess_x(xtest)
        
        return self.model.predict(xtest)
    
    def save(self, path):
        self.model.save(path)
        
    def _preprocess_x(self, x):
        x = self.textpreprocessor.preprocess(x)
        x = self.textpreprocessor.tokenize_dl(x, tokenizer_dl)
        x = self.textpreprocessor.pad_x(x, max_length)
        
        return x
    
    def _preprocess_y(self, y):
        y = pd.get_dummies(y)
        
        return y

#Second layer (CNN)
class CNN_Layer():
    def __init__(self, shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv1D(32, 1, padding = 'same', activation = 'relu', input_shape = shape))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        
        self.model.add(Conv1D(32, 2, padding = 'same', activation = 'relu'))
        self.model.add(MaxPooling1D(pool_size = 2))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(num_classes, activation = 'softmax'))

    def compile_model(self, loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy']):
        self.model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
        
    def fit(self, x, y):
        self.model.fit(x, y, epochs = 20, class_weight = {0: 1, 1: 1.5})
        
    def predict(self, xtest):
        return self.model.predict(xtest)
    
    def save(self, path):
        self.model.save(path)

#Ensemble class
class Ensemble_models():
    def __init__(self, sec_layer, first_layer):
        self.first_layer_models = first_layer
        self.sec_layer_model = sec_layer
        
    def generate_second_layer_x(self, x):
        new_x = np.zeros((x.shape[0], 8))
        curr_index = 0
        
        for model in self.first_layer_models:
            y_pred = model.predict(x)
            new_x[:, curr_index:(curr_index + y_pred.shape[1])] = y_pred
            curr_index = curr_index + y_pred.shape[1]
        
        new_x = np.reshape(new_x, new_x.shape + (1,))
        return new_x
    
    def fit(self, x, y):
        new_x = self.generate_second_layer_x(x)
        y = self._preprocess_y(y)
        
        self.sec_layer_model.compile_model()
        self.sec_layer_model.fit(new_x, y)
        
    
    def predict(self, xtest):
        y_mapper = {0: 'negative', 1: 'positive'}
        
        if type(xtest) == str:
            xtest = pd.Series(xtest)
        
        new_xtest = self.generate_second_layer_x(xtest)
        ypred = self.sec_layer_model.predict(new_xtest).argmax(axis = -1)
        
        return pd.Series(ypred).map(y_mapper)
    
    def save(self, path):
        self.sec_layer_model.save(path)
    
    def _preprocess_y(self, y):
        return pd.get_dummies(y)
