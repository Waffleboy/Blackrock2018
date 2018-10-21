import sys
import os
# new_path = './scripts'
# new_path2 = './Interest'
# if new_path not in sys.path:
# 	sys.path.append(new_path)
# if new_path2 not in sys.path:
# 	sys.path.append(new_path2)

import threading
import logging
from django.db import models
from django.contrib.postgres.fields import JSONField
import pandas as pd
import datetime
from collections import Counter,defaultdict
#from Ensemble import ensemble
#from relevant_pipeline import ensembleML
from apple.service_modules import priority_module
#from pipeline import predictUser
from django.utils import timezone
from apple.service_modules import twitter_user_downloader
from apple.service_modules import notable_account_finder

logger = logging.getLogger(__name__)

# Create your models here.

class TwitterUser(models.Model):
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)
	
	twitter_id = models.BigIntegerField(db_index=True) # Index to make searches quicker
	screen_name = models.CharField(max_length=250,default=None, blank=True, null=True)
	name = models.CharField(max_length=250,default=None, blank=True, null=True)
	profile_picture = models.TextField(default=None, blank=True, null=True)
	location = models.CharField(max_length=250,default=None, blank=True, null=True)
	description = models.TextField(default=None, blank=True, null=True)
	time_zone = models.CharField(max_length=100,default=None, blank=True, null=True)
	followers_count = models.BigIntegerField(default=None, blank=True, null=True)
	friends_count = models.BigIntegerField(default=None, blank=True, null=True)
	url = models.TextField(max_length=250,default=None, blank=True, null=True)

	age = models.CharField(max_length=250,default=None, blank=True, null=True)
	interest = models.CharField(max_length=250,default=None, blank=True, null=True)
	gender = models.CharField(max_length=250,default=None, blank=True, null=True)

	following_ids = JSONField(default=dict)
	raw_response = JSONField(default=dict)
	properties = JSONField(default=dict)

	def __str__(self):
		return "Twitter ID: {} screen_name: {}".format(self.twitter_id,self.screen_name)


	def find_or_create(user_response_dict):

		query = TwitterUser.objects.filter(twitter_id=user_response_dict['id'])
		if len(query) > 0:
			return query.first()

		#predictions = predictUser(user_response_dict['screen_name'])
		#interest = predictions[0]
		#age = predictions[1]
		#gender = predictions[2]

		#TODO: Abstract out to a info extractor class
		user_info = {'twitter_id': user_response_dict['id'],
					  'screen_name':user_response_dict['screen_name'],
					  'name':user_response_dict['name'],
					  'profile_picture':user_response_dict['profile_image_url_https'],
					 'location':user_response_dict['location'],
					 'description':user_response_dict['description'],
					 'time_zone':user_response_dict['time_zone'],
					 'followers_count':user_response_dict['followers_count'],
					 'friends_count':user_response_dict['friends_count'],
					 'url':user_response_dict['url']
					 #'raw_response':user_response_dict,
					 #'interest': interest,
					 #'age': age,
					 #'gender': gender
					 }

		#user_info['raw_response'] = user_info['raw_response']['_json']
		new_user = TwitterUser(**user_info)
		new_user.save()

		# start thread to populate ids following
		if new_user.following_ids == {}:
			t = threading.Thread(target=new_user.populate_following_ids)
			t.start()
		return new_user


	def populate_following_ids(self):
		ids = twitter_user_downloader.obtain_user_ids_of_friends(self.screen_name)
		self.following_ids["ids"] = ids
		return self.save()
	
	def find_notable_accounts(self):
		if self.following_ids != {}:
			ids = self.following_ids['ids']
			notable_accounts = notable_account_finder.convert_to_notable_account_format(ids)
			return notable_accounts
		return []



class Tweet(models.Model):
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)
	user = models.ForeignKey(TwitterUser,on_delete='cascade')

	tweet_id = models.BigIntegerField(db_index=True) # Index to make searches quicker
	text = models.TextField(max_length=300,default=None, blank=True, null=True)
	place = models.TextField(max_length=250,default=None, blank=True, null=True)
	latitude = models.DecimalField(max_digits=9, decimal_places=6,default=None, blank=True, null=True)
	longitude = models.DecimalField(max_digits=9, decimal_places=6,default=None, blank=True, null=True)
	retweet_count = models.BigIntegerField(default=None, blank=True, null=True)
	favorite_count = models.BigIntegerField(default=None, blank=True, null=True)

	priority = models.CharField(max_length=250,default=None, blank=True, null=True)
	sentiment = models.CharField(max_length=250,default=None, blank=True, null=True)
	resolved = models.BooleanField(default=False)
	resolved_time = models.DateTimeField(default=None, blank=True, null=True)

	raw_response = JSONField(default=dict)
	properties = JSONField(default=dict)

	def __str__(self):
		return "Tweet ID: {}".format(self.tweet_id)


	def find_or_create(tweet_dict):

		query = Tweet.objects.filter(tweet_id=tweet_dict['id'])
		if len(query) > 0:
			return query.first()

		#relevance = ensembleML.predict_x(tweet_dict['text']).iloc[0,]

		#if relevance == 'relevant':
		#	senti = ensemble.predict(tweet_dict['text'])[0]
		#else:
		#	senti = 'irrelevant'

		#TODO: Abstract out to a info extractor class
		tweet_info = {'tweet_id':tweet_dict['id'],
						'text':tweet_dict['text'],
						'place':tweet_dict['place'],
						'retweet_count':tweet_dict['retweet_count'],
						'favorite_count':tweet_dict['favorite_count']
						#'raw_response':tweet_dict,
						#'sentiment': senti
						}

		coordinates = tweet_dict['coordinates']
		if coordinates:
			tweet_info['longitude'] = coordinates['coordinates'][0]
			tweet_info['latitude'] = coordinates['coordinates'][1]\
		
		user = tweet_dict['user'].__dict__['_json']
		
		user = TwitterUser.find_or_create(user)

		tweet_info['user'] = user
		new_tweet = Tweet(**tweet_info)
		#priority_level = priority_module.get_priority_for_tweet(new_tweet)
		#new_tweet.priority = priority_level
		new_tweet.save()
		return new_tweet

	def resolve_tweet(self):
		self.resolved = True
		self.resolved_time = timezone.now() 
		return self.save()
