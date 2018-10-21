#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 22:05:44 2018

@author: thiru
"""

#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import os
import requests
import json
import logging

logger = logging.getLogger(__name__)

#Variables that contains the user credentials to access Twitter API
access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_ACCESS_SECRET"]
consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]

API_URL = "http://localhost:8000/streaming_api"

#This is a basic listener that just posts whatever given to the URL
class StdOutListener(StreamListener):

    def on_data(self, data):
        global API_URL
        data_json = json.loads(data)
        data_json["AUTH_KEY"] = os.environ["DJANGO_POST_KEY"]

        if "extended_tweet" in data_json.keys() and "full_text" in data_json['extended_tweet']:
            data_json["text"] = data_json['extended_tweet']["full_text"]
        res = requests.post(API_URL,json=data_json)
        return True

    def on_error(self, status):
        print (status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l, tweet_mode='extended')

    #This line filter Twitter Streams to capture data by the keywords
    stream.filter(track=['@applesupport'])



































