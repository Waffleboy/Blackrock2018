#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.getcwd()+'/')

from apple.models import *
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords

words_to_remove = set(['rt','apple','apple_support','applesupport','support'])
stop = set(stopwords.words('english'))
stop = stop.union(words_to_remove)

date_timelines = [1,7,30] #days
explained = ['daily','weekly','monthly']
dates = [datetime.date.today() - datetime.timedelta(days=x) for x in date_timelines]

def run():
    tweets = Tweet.objects.filter(created_at__gte = dates[-1]) # last
    for i in range(len(dates)-1,-1,-1):
        tweets = tweets.filter(created_at__gte = dates[i])

        text = [x.text for x in tweets]
        text = preprocess_text(text)
        text = ' '.join(text)
        if not text:
            text = 'No_Text'
        # lower max_font_size
        wordcloud = WordCloud(max_font_size=40,prefer_horizontal=0.98).generate(text)
        plt.figure(facecolor='k')
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.tight_layout(pad=0)
        plt.axis("off")
        name = explained[i] + '.png'
        plt.savefig('apple/static/wordclouds/{}'.format(name),bbox_inches='tight',facecolor='k')
    print("Wordcloud job complete!")


def preprocess_text(text_list):
    global words_to_remove
    text_list = [x.lower() for x in text_list]
    # strip urls
    text_list = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', x, flags=re.MULTILINE) for x in text_list]
    text_list = [' '.join([i for i in x.lower().split() if i not in stop]) for x in text_list]

    for word in words_to_remove:
        c = re.compile('(\s*){}(\s*)'.format(word))
        text_list = [c.sub('',x) for x in text_list]

    return text_list


if __name__ == '__main__':
    run()