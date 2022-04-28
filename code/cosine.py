import json
import csv
import re
import sys
import pandas as pd
import numpy as np

import gensim
from nltk.tokenize import word_tokenize

#####################################################
# Functions
def csv_loader(PATH):
    text = pd.read_csv(PATH, names=['review','sentiment']) 
    return text

#####################################################
args = sys.argv
if len(args) < 2:
    print("You forgot something")
FILE = args[1]  # name of interim csv file. For example: # games_dev
#ARG = args[2]    #

# Load Interim CSV file
data = csv_loader('../data/interim/' + FILE + '.csv')

# Split into X and Y (maybe not needed)
X, y = data[['review']], data[['sentiment']]

# Tokenize each review and lowercase everything
tknzd = []
for i in range(2): 
    row = X.iloc[i]['review']
    tokenized = [w.lower() for w in word_tokenize(row)]
    tknzd.append(tokenized)

# dictionary of tokens
dictionary = gensim.corpora.Dictionary(tknzd)

# make BOW
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in tknzd]
#print(len(tknzd[0:1][0]))
#print(len(corpus[0]))

# TFIDF to downplay frequent words
tf_idf = gensim.models.TfidfModel(corpus)
#for doc in tf_idf[corpus]:
#    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

# building the index
sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))

#





