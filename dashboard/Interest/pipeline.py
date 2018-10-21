# import libraries
import csv,re,random,tweepy,datetime
import pandas as pd
import numpy as np
import os
import pickle
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda
from keras.layers import LSTM
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from collections import Counter
import tensorflow as tf
from _pickle import dump, load
from nltk.corpus import stopwords
import requests
import time

import tensorflow as tf
# api keys
api_csv = pd.read_csv(r"Interest/apikeys.csv")
accesstokenlist=[]
for row in api_csv.iterrows():
    index, data = row
    accesstokenlist.append(data.tolist())

# pre-defined functions
def set_auth(currKeyList):
    auth = tweepy.auth.OAuthHandler(currKeyList[0], currKeyList[1])
    auth.set_access_token(currKeyList[2], currKeyList[3])
    api = tweepy.API(auth)
    return api

# extract tweets from a given screen name
def extractTweets(user,numOfTweets=500):
    ##Extract tweets from user (cycles api)
    currKeyIndex = 0
    currKeyList = accesstokenlist[currKeyIndex]
    totalKeys = len(accesstokenlist) #Number of twitter keys

    api = set_auth(currKeyList)

    def cycleKey():
        nonlocal currKeyIndex
        nonlocal currKeyList
        nonlocal totalKeys
        nonlocal api
        currKeyIndex = (currKeyIndex+1)%totalKeys
        currKeyList = accesstokenlist[currKeyIndex]
        api = set_auth(currKeyList)

    listOfTweets = []
    counter = numOfTweets // 200 #200 per request
    print('Extracting tweets from: ' + user)
    batch = api.user_timeline(screen_name = user,count=200)
    listOfTweets.extend(batch)
    listLen = listOfTweets[-1].id - 1
    while len(batch) > 0 and counter > 1:
        try:
            batch = api.user_timeline(screen_name = user,count=200,max_id=listLen)
            listOfTweets.extend(batch)
            listLen = listOfTweets[-1].id - 1
            counter -= 1
        except tweepy.TweepError:
            cycleKey()
            continue

    listOfTweets = [tweet.text for tweet in listOfTweets]
    return listOfTweets

# only for interest model
def processTweets(lst):
    for i in range(len(lst)):
        text = lst[i]
        text = text.lower()
        text = re.sub('RT @[\w_]+',"",text)
        text = re.sub('@[\w_]+','',text)
        text = re.sub(r"http\S+", "", text)
        lst[i] = text
    return lst

def has_default_photo(user):
    currKeyIndex = 0
    currKeyList = accesstokenlist[currKeyIndex]
    totalKeys = len(accesstokenlist)
    api = set_auth(currKeyList)
    res = api.lookup_users(screen_names=[user])
    default_photo = res[0].default_profile_image
    return default_photo

# Face API
# gets face API response
face_api = pd.read_csv(r"./Interest/faceAPI.csv")
keys = list(face_api.key)
keyNum = 0

def call_face_api(key, face_api_url, url):
    headers = { 'Ocp-Apim-Subscription-Key': key}
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender',
    }
    response = requests.post(face_api_url, params=params, headers=headers, json={"url": url})
    return response

def get_api_response(user):
    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
    url = 'https://avatars.io/twitter/' + user

    def keyNumShuffle(response):
        if response.status_code == 403: # max calls for key
            global keyNum
            keyNum += 1
        return min(keyNum, len(keys)-1)   # avoid keyNum > keys available

    try:
        response = call_face_api(keys[keyNum], face_api_url, url)
        if response.status_code == 429: # too many query for 26s
            time.sleep(20)
            response = call_face_api(key[keyNum], face_api_url, url)

        key = keyNumShuffle(response)
        response = response.json() # convert response obj to json

    except Exception as e:
        print(e)
        response = []

    return response

# takes in response object from face api
def get_face_data(response):
    face_info = response[0]
    json_res = face_info["faceAttributes"]
    age = json_res["age"]
    gender = json_res["gender"]
    return [age, gender]

# for ensembling ML models
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

    # for age prediction
    def format_to_age_label(self, list_of_labels):
        age_mapper = pd.Series(['Middle', 'Old', 'Teenager', 'Young Adult'])
        return age_mapper[list_of_labels]

    def predict_age(self, tweet):
        tweet = self.preprocess_x(tweet)
        tweet = self.tokenize_x(tweet)

        ypred = np.zeros((tweet.shape[0], self.num_class))

        for model in self.models:
            ypred = ypred + model.predict_proba(tweet)

        ypred = np.argmax(ypred, axis = -1)
        return self.format_to_age_label(ypred)

    # for gender prediction
    def format_to_gender_label(self, list_of_labels):
        gender_mapper = pd.Series(['Female', 'Male'])

        return gender_mapper[list_of_labels]

    def predict_gender(self, tweet):
        tweet = self.preprocess_x(tweet)
        tweet = self.tokenize_x(tweet)

        ypred = np.zeros((tweet.shape[0], self.num_class))

        for model in self.models:
            ypred = ypred + model.predict_proba(tweet)

        ypred = np.argmax(ypred, axis = -1)
        return self.format_to_gender_label(ypred)

# to decode prediction results
global predictionDic
predictionDic = {0:'Food',1:'Music',2:'News',3:'Politics',4:'Sports',5:'Tech',6:'Fashion',7:'Gaming',8:'Pets',9:'Reading',10:'Running',11:'Travel',12:'Volunteering'}
# helper functions for prediction
def predict_interest(tweets):
    interest_maxTweetLength = 141
    interest_data = processTweets(tweets)
    interest_data = interest_tokenizer.texts_to_sequences(interest_data)
    interest_data = sequence.pad_sequences(interest_data, maxlen = interest_maxTweetLength, padding = 'post')
    with graph.as_default():
        results = interest_model.predict(interest_data)
    results = np.argmax(results,axis=1)
    interest_key = Counter(results)
    interest_key = interest_key.most_common(1)[0][0]
    interest = predictionDic.get(interest_key)
    return interest

def predict_age(tweets):
    age_model = ensemble_ML(list_of_models = [age_lr, age_xgb, age_nb], \
                             vect = vect, num_class = 4)
    age_results = age_model.predict_age(tweets)
    predicted_age_count = Counter(age_results.index)
    predicted_age_key = predicted_age_count.most_common(1)[0][0]
    age = age_results[predicted_age_key].iloc[0]
    return age

def predict_gender(tweets):
    gender_model = ensemble_ML(list_of_models = [gender_lr, gender_xgb, gender_nb], \
                             vect = vect, num_class = 2)
    gender_results = gender_model.predict_gender(tweets)
    predicted_gender_count = Counter(gender_results.index)
    predicted_gender_key = predicted_gender_count.most_common(1)[0][0]
    gender = gender_results[predicted_gender_key].iloc[0]
    return gender

def stratify_age(age):
    if age <= 21:
        stratified = 'Teenager'
    elif age <= 30:
        stratified = 'Young Adult'
    elif age <= 50:
        stratified = 'Middle'
    else:
        stratified = 'Old'
    return stratified

# input: user screen name
# output: [interest, age, gender]
global interest_maxTweetLength
interest_maxTweetLength = 141

def predictUser(user,numTweets=500):
    # list of tweets
    userTweets = extractTweets(user,numTweets) # input number of tweets to pull as desired (>= 200)
    interest = predict_interest(userTweets)

    # predict age and gender by tweets
    if has_default_photo(user)==True:
        print("Default profile picture")
        age = predict_age(userTweets)
        gender = predict_gender(userTweets)

    # get age and gender from face api
    else:
        face_api_res = get_api_response(user)
        if len(face_api_res) == 1:
            print("1 face detected")
            face_data = get_face_data(face_api_res)
            age = face_data[0]
            age = stratify_age(age)
            gender = face_data[1].title()
        else:
            # 0 or multiple faces
            # predict age and gender by tweets
            print("0 or mutiple face detected")
            age = predict_age(userTweets)
            gender = predict_gender(userTweets)

    return [interest, age, gender]

# functions to load models
# load interest model
def load_cnn_interest():
    global interest_tokenizer
    global interest_embeddings_matrix
    global interest_model

    # .pickle files for interest model
    with open('./Interest/tokenizer30.pickle', 'rb') as handle:
        interest_tokenizer = pickle.load(handle)
    interest_tokenizer.oov_token = None
    with open('./Interest/embeddings30.pickle', 'rb') as handle:
        interest_embeddings_matrix = pickle.load(handle)
    interest_model = load_model('./Interest/bestmodel.h5')
    interest_model._make_predict_function()
    print("loaded interest model")

# load age model
def load_ML_age():
    global vect  # same vectorizer for gender
    global age_lr
    global age_xgb
    global age_nb

    # pkl files for age model
    vect = load(open('./Interest/vectorizer.pkl', 'rb'))
    age_lr = load(open('./Interest/age_lr.pkl', 'rb'))
    age_xgb = load(open('./Interest/age_xgb.pkl', 'rb'))
    age_nb = load(open('./Interest/age_nb.pkl', 'rb'))
    print("loaded age model")

# load age model
def load_ML_gender():
    global gender_lr
    global gender_xgb
    global gender_nb

    # pkl files for gender model
    gender_lr = load(open('./Interest/gender_lr.pkl', 'rb'))
    gender_xgb = load(open('./Interest/gender_xgb.pkl', 'rb'))
    gender_nb = load(open('./Interest/gender_nb.pkl', 'rb'))
    print("loaded gender model")

load_cnn_interest()
load_ML_age()
load_ML_gender()
global graph
graph = tf.get_default_graph()
