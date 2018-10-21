from apple.models import *

def get_priority_for_tweet(tweet_obj):
	user = tweet_obj.user
	follower_count = user.followers_count
	return popularity_identify_v0(follower_count)


def popularity_identify_v0(follower_count):
	if follower_count >= 1000:
		return 'High'
	elif follower_count >= 500:
		return 'Med'
	return 'Low'