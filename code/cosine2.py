import json
import csv
import re
import sys
import pandas as pd
import numpy as np
import heapq

import gensim
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#####################################################
# Functions
def csv_loader(PATH):
    text = pd.read_csv(PATH, names=['review','sentiment']) 
    return text

#####################################################
args = sys.argv
if len(args) < 2:
    print("You forgot something")
FILE_1 = args[1]  # name of interim csv file. For example: # games_train
FILE_2 = args[2]  # name of comparison interim csv file. For example: # sew_train
N_DIS = int(args[3])   # number of dissimilar embeddings to select
# python3 cosine.py 'music_dev' 'music_dev' 2

# Load Interim CSV file and split into X and y
data_1 = csv_loader('../data/interim/' + FILE_1 + '.csv')
X_1, y_1 = data_1[['review']], data_1[['sentiment']]

# make the corpus a list of strings
corp_1 = []
for i in range(5): 
    row = X_1.iloc[i]['review']
    corp_1.append(row)

print(f"First Corpus: {corp_1[0]}")

# Use TFIDF to lower impact of frequent words
count_vectorizer = TfidfVectorizer(stop_words='english')
sparse_matrix = count_vectorizer.fit_transform(corp_1)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names_out())
#print(df.head)

#############################################################
# Load Second Interim CSV file and split into X and y
data_2 = csv_loader('../data/interim/' + FILE_2 + '.csv')
X_2, y_2 = data_2[['review']], data_2[['sentiment']]

corp_2 = []
for i in range(5): 
    row = X_2.iloc[i]['review']
    corp_2.append(row)

#########################################
# compute cosine similarity
co_sim = cosine_similarity(df, df)

avg_sims = [] # array of averages
for i in range(len(corp_2): 
    avg_sims.append(("thing",i))    

print(f"\nSecond corpus: {corp_2[0]}")

print("\nNumber of documents to compare:",len(corp_2))  
print(f"\nAverage similarities: {avg_sims}")

########################################
# Find most dissimilar
pq = heapq.nsmallest(N_DIS, avg_sims, key=None)
print(f"\nPriority Q: {pq})")

most_dis = []
for tup in pq:
    most_dis.append(X_2.iloc[tup[1]]['review'])
print(f"\nMost Dissimilar Sentences: {most_dis}")


