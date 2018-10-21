# Create your views here.
import logging
import random, string
import sys
import json
import os
from threading import Thread
import datetime

from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from apple.models import Tweet,TwitterUser
from apple import views_helper 

from apple.forms import replyForm
#from apple.tweet_reply import update_stat
from apple.customer_helper import get_user_info, get_user_tweets
#from apple.service_modules.autoreply_module import autoreply,autoreply_all

import logging
logger = logging.getLogger(__name__)

@login_required(login_url="login/")
def index(request):
	context = {}

	# get:
	# 1) daily / weekly / monthly wordcloud info. this should be a pregenerated image, not run on demand
	# 2) topic modelling, same.

	tweets = Tweet.objects.all()
	positive_tweets = tweets.filter(sentiment='positive')
	negative_tweets = tweets.filter(sentiment='negative')
	yesterday = datetime.datetime.today() - datetime.timedelta(days=1)
	hour_ago = datetime.datetime.today() - datetime.timedelta(minutes=60)

	context['open_positive_tweets_count'] = positive_tweets.filter(resolved=False).count()
	context['open_negative_tweets_count'] = negative_tweets.filter(resolved=False).count()

	twenty_four_hour_negative_tweets = negative_tweets.filter(created_at__gte = yesterday)
	high_priority_negative_tweets = twenty_four_hour_negative_tweets.filter(priority='High')

	twenty_four_hour_negative_tweets_count = twenty_four_hour_negative_tweets.count()
	high_priority_negative_tweets_count = high_priority_negative_tweets.count()

	context["negative_tweet_24hr_count"] = twenty_four_hour_negative_tweets_count
	context["negative_high_priority_count_24hr_count"] =  high_priority_negative_tweets_count

	if high_priority_negative_tweets_count != 0:
		negative_high_priority_percentage = round((high_priority_negative_tweets_count / twenty_four_hour_negative_tweets_count) * 100 ,2) 
	else:
		negative_high_priority_percentage = 0

	context["negative_high_priority_percentage"] = negative_high_priority_percentage

	hourly_negative = negative_tweets.filter(created_at__gte = hour_ago)
	hourly_positive = positive_tweets.filter(created_at__gte = hour_ago)
	context["tweet_velocity"] = hourly_negative.count() + hourly_positive.count()
	context["resolve_velocity"] = sum([1 for x in hourly_negative if x.resolved_time]) + sum([1 for x in hourly_positive if x.resolved_time])

	context['positive_tweets'] = views_helper.format_pos_tweets_to_table(positive_tweets)
	context['negative_tweets'] = views_helper.format_neg_tweets_to_table(negative_tweets)

	context['user'] = request.user
	return render(request, 'dashboard.html', context)

@login_required(login_url="login/")
def customers(request):
	context = {}

	all_users = TwitterUser.objects.all()
	user_info = get_user_info(all_users)

	context['users'] = user_info
	return render(request, 'customers.html', context)

@login_required(login_url="login/")
def profile(request):
	context = {'redirected': 0}
	
	if request.method == "GET":
		request_dict = list(dict(request.GET).keys())
		form = replyForm()

		if len(request_dict) != 0:
			dict_content = request_dict[0].split('?')
			screenname = dict_content[0]
			user = TwitterUser.objects.get(screen_name = screenname)
			if len(dict_content) == 2:
				tweet_id = dict_content[1]

				tweet = Tweet.objects.get(tweet_id = tweet_id)
				

				context['tweet_id'] = tweet.tweet_id
				context['tweet'] = tweet.text

				context['form'] = form
			else:
				context['tweets'] = get_user_tweets(user)
				context['redirected'] = 1

	else:
		screenname = request.POST['screen_name']
		user = TwitterUser.objects.get(screen_name = screenname)

		if 'tweet_id' in request.POST.keys():
			form = replyForm(request.POST)

			tweet_id = request.POST['tweet_id']
			tweet = Tweet.objects.get(tweet_id = tweet_id)
			tweet.resolve_tweet()

			if form.is_valid():
				reply = form.cleaned_data['reply']
				update_stat(screenname, reply, tweet_id)

				form = replyForm()

		context['tweets'] = get_user_tweets(user)
		context['redirected'] = 1

	#To hide errors from multiple get requests when loading new page
	try:
		context['profile_pic'] = user.profile_picture
		context['screen_name'] = user.screen_name
		context['followers'] = user.followers_count
		context['friends'] = user.friends_count
		context['interest'] = user.interest
		context['notable_accounts'] = user.find_notable_accounts()
		
		context['age'] = user.age
		context['gender'] = user.gender
	except:
		pass
	return render(request, 'profile.html', context)


@login_required(login_url="login/")
def insights(request):
	context = {}
	return render(request, 'chartjs.html', context)

@csrf_exempt
def resolve_api_post(request):
	if request.method == 'POST':
		tweet_id = request.POST['tweet_id']
		tweet = Tweet.objects.filter(tweet_id = int(tweet_id)).first()
		if not tweet:
			logger.warn("Invalid tweet ID for {}".format(tweet_id))
			return HttpResponse("Invalid tweet ID")
		tweet.resolve_tweet()
		response = {'Resolved':True,
					'tweet_id':'{}'.format(tweet_id)}
		return HttpResponse(json.dumps(response))
	return HttpResponse("Invalid operands")

##TODO: Move to seperate thread
@csrf_exempt 
def stream_api_post(request):
	if request.method == 'POST':
		logger.info("Got a stream API post")
		tweet_details = json.loads(request.body)
		if not tweet_details["AUTH_KEY"] == os.environ["DJANGO_POST_KEY"]:
			return HttpResponse("Invalid Auth key")

		tweet = Tweet.find_or_create(tweet_details)
		response = {'status':'fail'}

		if tweet:
			response['status'] = 'success'
			response['tweet_id'] = tweet.tweet_id
			response['user_id'] = tweet.user.twitter_id
			response['screen_name'] = tweet.user.screen_name
		
		return HttpResponse(json.dumps(response))

	else: #GET
		return HttpResponse("Invalid operands")

@csrf_exempt
def autoreply_api(request):
	if request.method == 'POST':
		tweet_id = request.POST['tweet_id']
		autoreply_all = request.POST["all"]
		if tweet_id:
			response = autoreply(tweet_id)
		elif autoreply_all:
			response = autoreply_all()
		
	return HttpResponse(json.dumps(response))


@csrf_exempt
def misclassified_api(request):
	if request.method == 'POST':
		tweet_id = request.POST['tweet_id']
		tweet = Tweet.objects.filter(tweet_id = tweet_id).first()
		mapper = {"positive":"negative","negative":"positive"}

		response = {"success":False}

		if tweet:
			sentiment = tweet.sentiment
			tweet.properties["misclassified"] = {"status":True,"original":sentiment}

			if sentiment == 'irrelevant':
				correct = request.POST['correct']
				tweet.sentiment = correct

			else: #pos or neg
				other_sentiment = mapper[sentiment]
				tweet.sentiment = other_sentiment
			tweet.save()

			response = {"success":True,"tweet_id":tweet_id,
					"new_sentiment": tweet.sentiment,
					"old_sentiment":tweet.properties["misclassified"]["original"]}
		
	return HttpResponse(json.dumps(response))