# imports
import pandas as pd
import re

# line=True fixes trailing error
data_train = pd.read_json('../data/raw/music_reviews_train.json', lines=True)
data_dev = pd.read_json('../data/raw/music_reviews_dev.json', lines=True)
data_test_masked = pd.read_json('../data/raw/music_reviews_test_masked.json', lines=True)

def tokenizer(sentence):
    tok = re.compile('[\'\"]|[A-Za-z]+|[.?!:\'\"]+')   
    return tok.findall(sentence)

def prep(df):
    df['concatSummaryReview'] = df['summary'] + ' ' + df['reviewText']
    df['concatSummaryReview'] = df['concatSummaryReview']
    df['concatSummaryReview'] = df['concatSummaryReview'].str.lower().fillna('<NA>')
    X = df['concatSummaryReview'].apply(tokenizer)
    y = df['sentiment']
    return X, y