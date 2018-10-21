#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 15:03:12 2018

@author: thiru
"""

import tweepy
import pandas as pd
import logging
import json
import time
import itertools
from sklearn.externals import joblib
import os, csv
import numpy as np

## Important assumption - csv in format consumerkey/consumersecret/accesskey/accesssecret
TWITTER_KEYS_PATH = 'Interest/apikeys.csv'
logger = logging.getLogger(__name__)
current_key_idx = 0
api = None # will be set later

#==============================================================================
#                               main functions
#==============================================================================

# Get a csv of twitter IDs that follow a twitter account
# Input:
# 1) <string> twitter handle name
# 2) <string> filename to save to
def obtain_user_ids_of_followers(twitter_handle, filename, resume = True):
    global current_key_idx, TWITTER_KEYS_PATH
    endpoint = 'followers/followers/ids'
    key_df = load_and_verify_keys(TWITTER_KEYS_PATH)

    create_or_override_file(filename,resume,['current_user','twitter_ids','cursor'])
    current_cursor = -1
    counter = 0

    # get latest page num
    if resume and os.path.exists(filename):
        current_cursor = get_cursor_position(filename)

    with open(filename,'a') as f:
        writer = csv.writer(f)

        while current_cursor or current_cursor == -1:
            if counter % 50 == 0:
                print("On page {}".format(counter))
            api = check_and_switch_key_if_needed(key_df,endpoint)
            counter += 1
            user_id_tup = api.followers_ids(twitter_handle,cursor = current_cursor)
            current_cursor = user_id_tup[1][1]
            user_ids = user_id_tup[0]
            for id_ in user_ids:
                if id_ == user_ids[-1]: # if last id
                    writer.writerow([twitter_handle,id_,current_cursor])
                    break
                writer.writerow([twitter_handle,id_,''])

    return

def obtain_user_ids_of_friends(twitter_handle):
    global current_key_idx, TWITTER_KEYS_PATH
    endpoint = 'friends/friends/ids'
    key_df = load_and_verify_keys(TWITTER_KEYS_PATH)
    ids = []
    api = check_and_switch_key_if_needed(key_df,endpoint)
    for page in tweepy.Cursor(api.friends_ids, screen_name=twitter_handle).pages():
        api = check_and_switch_key_if_needed(key_df,endpoint)
        ids.extend(page)

        if len(ids) >= 10000:
            break

    return ids

# Get full account details of all users that follow a twitter accont in JSON form
# Input: 
# 1) CSV of twitter IDS (from obtain_user_ids)
# 2) filename to save to 
def get_user_details(path_to_csv, output_json_filename,resume = True):
    endpoint = 'users/users/lookup'
    global current_key_idx, TWITTER_KEYS_PATH
    key_df = load_and_verify_keys(TWITTER_KEYS_PATH)
    id_df = pd.read_csv(path_to_csv)
    batch_size = 100 #twitter max 100 users / req
    id_batches = split_column_to_batches(id_df,'twitter_ids',batch_size)
    print("Total batches: {}".format(len(id_batches)))
    failed_batch_ids = []
    resume_point = 0
    if resume:
        resume_point = get_resume_point(output_json_filename,batch_size)

    for i in range(resume_point,len(id_batches)):
        if i % 50 == 0:
            print("Currently at batch {}".format(i))
        
        try:
            scrape_and_dump(key_df,id_batches,output_json_filename,i,endpoint)
        except Exception as e:
            try:
                print(e)
                print("Exception encountered for batch {}. Trying again after 10 mins..".format(i))
                time.sleep(10*60)
                scrape_and_dump(key_df,id_batches,output_json_filename,i,endpoint)
            except Exception as e:
                print(e)
                print("Failed again for batch {}. Waiting 10 minutes and skipping batch..".format(i))
                failed_batch_ids.append(i)
                continue
            
    fix_to_proper_json(output_json_filename) #in the process of scraping its not a proper json
    print("Done!")
    return

def scrape_and_dump(key_df,id_batches,output_json_filename,i,endpoint):
    api = check_and_switch_key_if_needed(key_df,endpoint)
    current_batch = id_batches[i]
    user_objs = api.lookup_users(current_batch)
    user_objs_json = [x._json for x in user_objs]
    add_to_json(user_objs_json,output_json_filename)
    
#==============================================================================
#                               helpers
#==============================================================================
def get_cursor_position(filename):
    df = pd.read_csv(filename)
    if len(df['cursor']) > 0:
        current_cursor = df['cursor'].tail(1).values[0]
    return current_cursor

def get_resume_point(output_json_filename,batch_size):
    if os.path.exists(output_json_filename):
        data = read_incorrect_json_to_list(output_json_filename)
        return len(data)
    return 0

def split_column_to_batches(df,colname,n):
    grpby = df[colname].groupby(np.arange(len(df)) // n)
    chunks = [x[1].values.tolist() for x in grpby]
    return chunks

def generate_csv(filename,column_names):
    df = pd.DataFrame(columns=column_names)
    df.to_csv(filename,index=False)
    return

def create_or_override_file(filename,resume,cols):
    if not os.path.exists(filename) or resume == False:
        if resume == False:
            logger.warn("Overwriting existing file..")
        generate_csv(filename,cols)
    return

def add_to_json(list_of_json,filename):
    if os.path.exists(filename):
        with open(filename, 'a') as f:
            f.write(','+os.linesep)
    with open(filename, 'a') as f:
        json.dump(list_of_json, f)

def read_incorrect_json_to_list(filename):
    data = open(filename).read()
    data = data.split(',\n')
    data = [json.loads(x) for x in data]
    return data

def fix_to_proper_json(filename):
    data = read_incorrect_json_to_list(filename)
    data_flattened = list(itertools.chain(*data))
    with open(filename, 'w') as f:
        json.dump(data_flattened, f)


#==============================================================================
#                       Key Cycling Logic
#==============================================================================
def check_and_switch_key_if_needed(df,endpoint):
    global api, current_key_idx
    if not api:
       api = load_key(0,df)

    rate_limits = api.rate_limit_status()

    main_endpoint = endpoint[:endpoint.find('/')]
    sub_endpoint = endpoint[endpoint.find('/'):]

    if rate_limits['resources'][main_endpoint][sub_endpoint]['remaining'] > 1:
        return api

    cycled_count = 0
    if main_endpoint not in rate_limits['resources']:
        logger.warning("WARNING! Incorrect endpoint specified. Cycling will succeed but failure might occur")
        api = cycle_key(df)
        return api

    while rate_limits['resources'][main_endpoint][sub_endpoint]['remaining'] <= 1:
        if cycled_count >= len(df):
            logger.warning("All keys exhausted - waiting for 10 minutes.")
            logger.warning("Current key idx: {}".format(current_key_idx))
            time.sleep(60*10)

        api = cycle_key(df)
        rate_limits = api.rate_limit_status()
        cycled_count += 1
    logger.warning("Key switched - Current key_idx: {}".format(current_key_idx))
    return api

def load_and_verify_keys(TWITTER_KEYS_PATH):
    df = pd.read_csv(TWITTER_KEYS_PATH)
    if len(df.columns) != 4:
        logger.error("CSV does not have 4 columns for the access tokens")
    return df

def load_key(number,df):
    col_names = df.columns
    auth = tweepy.auth.OAuthHandler(df[col_names[0]][number], df[col_names[1]][number])
    auth.set_access_token(df[col_names[2]][number], df[col_names[3]][number])
    api = tweepy.API(auth)
    return api

def cycle_key(df):
    global current_key_idx
    current_key_idx = get_next_key(df)
    api = load_key(current_key_idx,df)
    return api

def get_next_key(df):
    global current_key_idx
    return (current_key_idx+1) % len(df)

