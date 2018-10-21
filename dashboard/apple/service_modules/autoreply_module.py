from apple.models import *
import random
from apple import tweet_reply

positive_templates = ["Hey {}, we appreciate that, thank you for being a loyal supporter!",
						"We really appreciate that, thank you {}!",
						"Thanks for your positive feedback {}! We'll continue to strive forward :)",
						"Right on - Thanks {}!"]

# currently only meant to be used for positive tweets
def autoreply_all():
	tweets = Tweet.objects.filter(sentiment = 'positive',resolved = False)
	success = 0
	num = len(tweets)
	failed = []

	for tweet in tweets:
		res = autoreply_given_tweet(tweet)
		if res["success"] == 'true':
			success += 1
		else:
			failed.append(tweet.tweet_id)

	return {"success_count": success, "total":num,"failed":failed}

def autoreply(tweet_id):
	tweet = Tweet.objects.filter(tweet_id = tweet_id).first()
	return autoreply_given_tweet(tweet)


def autoreply_given_tweet(tweet):
	if tweet:
		sentiment = tweet.sentiment
		screenname = tweet.user.screen_name
		response = 'Sentiment not generated'
		if sentiment == 'positive':
			response = generate_positive_tweet(screenname)
		elif sentiment == 'negative':
			response = generate_negative_tweet(screenname)

	result = tweet_reply.update_stat(screenname,response,tweet.tweet_id)
	res = {"success":"false"}
	if result:
		res["success"] = "true"
		res["result_debug"] = result._json
		tweet.resolve_tweet()
	return res


def generate_positive_tweet(screenname):
	return random.choice(positive_templates).format(screenname)

def generate_negative_tweet(screenname):
	## Gather information about user and tweet. Whats the issue that its negative?

	## If influential || serious
		#find brands they like relating to that interest
	return "Not done"
