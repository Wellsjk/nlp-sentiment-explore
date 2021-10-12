######################
# Sentiment Analysis #
######################

'''
testing out of the box NLP libraries for sentiment analysis on comments 
script is set up to take comments as rows (df) and output sentiment score for each row
potential libraries
 - VADER
 - blobtext
 - custom NLTK?
 '''

import pandas as pd
import numpy as np

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#### Import Data ####
# import data to df from csv
df = pd.read_csv('data/IMDB Dataset.csv') # testing with imdb reviews
df['sentiment'] = np.where(df['sentiment']=='positive', 1, 0)
df = df[:500]
print(df)


#### comment sentiment ####
analyzer = SentimentIntensityAnalyzer() # initialize vader

text_blob_sentiment = []
vader_sentiment = []

for comment in df['review']:
    text_blob_sentiment.append(TextBlob(comment).sentiment.polarity)
    vader_sentiment.append(analyzer.polarity_scores(comment)['compound'])



df['tb_sentiment'] = text_blob_sentiment
df['v_sentiment'] = vader_sentiment


print(df['tb_sentiment'].describe())
print(df['tb_sentiment'])


print(df['v_sentiment'].describe())
print(df['v_sentiment'])

