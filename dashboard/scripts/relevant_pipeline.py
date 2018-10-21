
from _pickle import load

import pandas as pd
import numpy as np
from nltk.corpus import stopwords

vect = load(open('./scripts/vect_relevant.pkl', 'rb'))
lr = load(open('./scripts/lr_relevant.pkl', 'rb'))
xgb = load(open('./scripts/xgb_relevant.pkl', 'rb'))
nb = load(open('./scripts/nb_relevant.pkl', 'rb'))

class ensemble_ML():
    def __init__(self, list_of_models, vect, num_class):
        self.models = list_of_models
        self.vect = vect
        self.num_class = num_class
        
    def preprocess_x(self, tweet):
        tweets = pd.Series(tweet)
        
        #convert tweets to lower case
        tweets = tweets.apply(str.lower)
        #replaces any urls with URL
        tweets = tweets.str.replace(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ')
        #removes user mentions
        tweets = tweets.str.replace(r"@\S+", " MENTIONS ")
        #removes hashtags
        tweets = tweets.str.replace(r"#\S+", " HASHTAGS ")
        #removes rt
        tweets = tweets.str.replace(r'\brt\b', ' RETWEETS ')
        #removes multiple white spaces 
        tweets = tweets.str.replace(r'\s+', " ")
        #strips both ends of white space
        tweets = tweets.apply(str.strip)
        
        #loading stopwords
        stop = stopwords.words('english')
        
        #removing stopwords
        tweets = tweets.str.split()
        tweets = tweets.apply(lambda x: [item for item in x if item not in stop])
        tweets = tweets.str.join(" ")
        
        return tweets
    
    def tokenize_x(self, tweet):
        tweet = vect.transform(tweet)
        
        return tweet

    def format_to_label(self, list_of_labels):
        ymapper = pd.Series(['irrelevant', 'relevant'])
        
        return ymapper[list_of_labels]
        
    def predict_x(self, tweet):
        tweet = self.preprocess_x(tweet)
        tweet = self.tokenize_x(tweet)
        
        ypred = np.zeros((tweet.shape[0], self.num_class))
        
        for model in self.models:
            ypred = ypred + model.predict_proba(tweet)
        
        ypred = np.argmax(ypred, axis = -1)
        return self.format_to_label(ypred)

ensembleML = ensemble_ML(list_of_models = [lr, xgb, nb], vect = vect, num_class = 2)





