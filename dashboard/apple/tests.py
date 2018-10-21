from django.test import TestCase

# Create your tests here.

import json
from django.core import serializers
from apple.models import *

## Test save a model to database

#import code; code.interact(local=dict(globals(), **locals()))

class ModelTestCase(TestCase):
	data = None

	def setUp(self):
		global data
		with open("test_data/test_response_data.json") as f:
			data = json.load(f)

	def test_save_user(self):
		result = TwitterUser.find_or_create(data['user'])
		return self.assertEqual(14065548,result.twitter_id)

	def test_save_tweet(self):
		result = Tweet.find_or_create(data)
		return self.assertEqual(978273515135819776,result.tweet_id)

	def test_certain_fields_exist(self):
		result = Tweet.find_or_create(data)
		return self.assertEqual(result.priority,'High')
    	
