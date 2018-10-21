import os

from tweepy import OAuthHandler
from tweepy import API

access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_ACCESS_SECRET"]
consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = API(auth)

def update_stat(screenname, reply, tweet_id):
	global api
	return api.update_status('@' + screenname + ' ' + reply, in_reply_to_status_id = tweet_id)