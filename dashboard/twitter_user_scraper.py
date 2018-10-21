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


user_lst = ["realdonaldtrump","jimcramer","TheStalwart",'TruthGundlach','Carl_C_Icahn',
                'ReformedBroker','bespokeinvest']


import tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
#stuff = api.user_timeline(screen_name = 'danieltosh', count = 100, include_rts = True)


tweet_ids_to_scrape = [1009895079631351808]#959474287580143617,1052667064064851971,1053280005395042305]

tweets = api.statuses_lookup(tweet_ids_to_scrape) # id_list is the list of tweet ids

from apple.models import Tweet
for entry in tweets:
	Tweet.find_or_create(entry.__dict__)