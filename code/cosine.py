import json
import csv
import re
import sys
import pandas as pd
import numpy as np
import heapq

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
FILE_1 = args[1]  # name of interim csv file. For example: # games_train
FILE_2 = args[2]  # name of comparison interim csv file. For example: # sew_train
N_DIS = int(args[3])   # number of dissimilar embeddings to select
# python3 cosine.py 'music_dev' 'sew_val' 2

# Load Interim CSV file and split into X and y
data_1 = csv_loader('../data/interim/' + FILE_1 + '.csv')
X_1, y_1 = data_1[['review']], data_1[['sentiment']]

# Tokenize each review and lowercase everything
corp_1 = []
for i in range(1,2): 
    row = X_1.iloc[i]['review']
    tokenized = [w.lower() for w in word_tokenize(row)]
    corp_1.append(tokenized)

print(f"First Corpus: {corp_1[0]}")

# dictionary of tokens
dictionary = gensim.corpora.Dictionary(corp_1)

# make BOW
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in corp_1]

# TFIDF to downplay frequent words
tf_idf = gensim.models.TfidfModel(corpus)
#for doc in tf_idf[corpus]:
#    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

# building the index
sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))

#############################################################
# Load Second Interim CSV file and split into X and y
data_2 = csv_loader('../data/interim/' + FILE_2 + '.csv')
X_2, y_2 = data_2[['review']], data_2[['sentiment']]

corp_2 = []
avg_sims = [] # array of averages
for i in range(1,2): 
    row = X_2.iloc[i]['review']
    tokenized = [w.lower() for w in word_tokenize(row)] # Tokenize each review and lowercase everything
    query_doc_bow = dictionary.doc2bow(tokenized) # update an existing dictionary and create bag of words
    corp_2.append(tokenized)
    
    # perform a similarity query against the corpus
    query_doc_tf_idf = tf_idf[query_doc_bow]
    
    doc_sim = sims[query_doc_tf_idf]
    # print(document_number, document_similarity)
    #print('Comparing Result:', doc_sim)
    
    sum_of_sims =(np.sum(doc_sim, dtype=np.float32))
    sim_ave = round(sum_of_sims/len(corp_1), 2)
    avg_sims.append((sim_ave,i))
    
    #print(f"Average Similarity: {sim_ave}")
    

print(f"Second corpus: {corp_2[0]}")

print("Number of documents to compare:",len(corp_2))  
print(f"Average similarities: {avg_sims}")

pq = heapq.nsmallest(N_DIS, avg_sims, key=None)
print(f"Priority Q: {pq})")

most_dis = []
for tup in pq:
    most_dis.append(X_2.iloc[tup[1]]['review'])
print(f"Most Dissimilar Sentences: {most_dis}")


