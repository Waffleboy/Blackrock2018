#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:22:45 2018

@author: thiru
"""

from nltk.corpus import stopwords
from apple.models import *
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import pandas as pd
import datetime
import re
import pyLDAvis.gensim as gensimvis
import pyLDAvis


words_to_remove = set(['rt','apple','apple_support','applesupport','support'])
stop = set(stopwords.words('english'))
stop = stop.union(words_to_remove)

exclude = set(string.punctuation)
lemma = WordNetLemmatizer()

date_timelines = [1,7,30] #days
explained = ['daily','weekly','monthly']
dates = [datetime.date.today() - datetime.timedelta(days=x) for x in date_timelines]


def run():
    tweets = Tweet.objects.filter(created_at__gte = dates[-1]) # last
    for i in range(len(dates)-1,-1,-1):
        tweets = tweets.filter(created_at__gte = dates[i])
        num_topics = 6

        text = [x.text for x in tweets]
        preprocessed_text = preprocess_text(text)
        texts_cleaned = [(clean(x)).split() for x in preprocessed_text]
        if len(texts_cleaned) == 0:
            texts_cleaned = ['no text'.split()]
        texts_cleaned = pd.Series(texts_cleaned)


        ldamodel,dictionary,doc_term_matrix  = run_lda(texts_cleaned,num_topics)

#        results = pretty_print_results(ldamodel,num_topics,num_words = 7)
#        for entry in results:
#            print(entry)
#        print('\n')

        vis_data = gensimvis.prepare(ldamodel,doc_term_matrix, dictionary )
        pyLDAvis.save_html(vis_data,'apple/static/ldatopics/{}.html'.format(explained[i]))

    print("LDA job complete!")



def clean(doc):
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


def run_lda(doc_clean,num_topics):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    lda = gensim.models.ldamodel.LdaModel
    ldamodel = lda(doc_term_matrix,num_topics=num_topics,id2word = dictionary,passes = 60)
    return ldamodel,dictionary,doc_term_matrix

def pretty_print_results(ldamodel,num_topics,num_words):
    z = ldamodel.print_topics(num_topics=num_topics,num_words=num_words)
    group = 1
    for i in range(len(z)):
        entry = z[i][1]
        entry = ''.join([i for i in entry if not (i.isdigit() or i == '.' or i == '*')])
        entry = entry.replace('"','')
        entry = '{}. {}'.format(group,entry)
        entry = entry.replace(' + ',', ')
        z[i] = entry
        group += 1
    return z

def preprocess_text(text_list):
    words_to_remove = ['rt']
    text_list = [x.lower() for x in text_list]
    # strip urls
    text_list = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', x, flags=re.MULTILINE) for x in text_list]
    text_list = [' '.join([i for i in x.lower().split() if i not in stop]) for x in text_list]

    for word in words_to_remove:
        c = re.compile('(\s*){}(\s*)'.format(word))
        text_list = [c.sub('',x) for x in text_list]

    return text_list
