

def format_pos_tweets_to_table(tweet_query_set):
	return [[x.created_at.strftime("%d/%m /%Y %M:%S"),x.user.screen_name,x.text, x.tweet_id] for x in tweet_query_set if x.resolved == False]

def format_neg_tweets_to_table(tweet_query_set):
	return [[x.created_at.strftime("%d/%m /%Y %M:%S"),x.user.screen_name,x.text, x.tweet_id] for x in tweet_query_set if x.resolved == False]

def format_other_tweets_to_table(tweet_query_set):
	return [[x.created_at.strftime("%d/%m /%Y %M:%S"),x.user.screen_name,x.text,x.tweet_id] for x in tweet_query_set]
	