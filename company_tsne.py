#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 15:04:00 2018

@author: waffleboy
"""

import pandas as pd
import numpy as np
twitter_list = ['@jimcramer',"@TheStalwart",'@TruthGundlach','@Carl_C_Icahn',
                '@ReformedBroker','@benbernanke','@bespokeinvest']


df = pd.read_csv("/storage/git/Blackrock2018/datafiles/stock_price_sector.csv")
remove_columns = ['None','dimension','ticker','datekey','calendardate',\
                  'reportperiod','lastupdated']

select_columns = ['ticker','sector','assets','cashneq','currentratio','debt','ebitda',\
                  'inventory','liabilities','marketcap','netmargin','pe']


# missing values
#issing = np.sum(pd.isnull(df),axis=0)
#missing = list(missing[missing>2].index)
#df = df.drop(missing,axis=1)
#df = df.dropna()


df = df[df['reportperiod'] > '2018-01-01']
df = df[df["dimension"] == "ART"]

# experiment - take last date only
df = df.drop_duplicates(keep='last',subset='ticker')

import random

colourmap = {k:(random.random(),random.random(),random.random()) for k in sectors}

df = df[select_columns]


#df = df.drop(remove_columns,axis=1)



from sklearn.manifold import TSNE
df = df.dropna()

classes = df["ticker"]
sectors = df['sector']

df = df.drop(['ticker','sector'],axis=1)
df = df.sub(df.mean(axis=0), axis=1)

X_embedded = TSNE(n_components=2).fit_transform(df)

import matplotlib.pyplot as plt

x = pd.DataFrame(X_embedded)
x["label"] = sectors.values
x["classes"] = classes.values

fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('x', fontsize = 15)
ax.set_ylabel('y', fontsize = 15)
ax.set_title('2 component tsne', fontsize = 20)

for entry in x['label'].unique():
    sub = x[x["label"] == entry]
    plt.scatter(sub[0],sub[1],label=entry,color=colourmap[entry])
    for i in sub.index:
        plt.annotate(x["classes"].iloc[i],(sub[0][i], sub[1][i]))
#    
#for i,entry in enumerate(X_embedded):
#    l = sectors.iloc[i]
#    l = sectors.iloc[i]
#    plt.annotate(classes.iloc[i],(X_embedded[i][0], X_embedded[i][1]))
plt.legend()
plt.show()
#
#
#
#
#
#
#import quandl,os
#quandl.ApiConfig.api_key =os.environ["QUANDL_API_KEY"]
#
#
#ta = quandl.get_table('IFT/NSA',ticker=('AAPL','MSFT'))#, date='2014-01-01', brand_ticker='MCD')
