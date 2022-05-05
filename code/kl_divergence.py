import json
import csv
import re
import sys
import pandas as pd
import numpy as np
import heapq
import sys
from tqdm import tqdm
import time
import gensim

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

start_time = time.time()

# Functions
def csv_loader(PATH):
    text = pd.read_csv(PATH, names=['review','sentiment']) 
    return text

args = sys.argv
if len(args) < 2:
    print("You forgot something")
FILE_1 = args[1]        # name of base corpus. For example: # music_train
FILE_2 = args[2]        # name of corpus file to select training samples. For example: # sew_train or games_train
FILE_NAME = args[3]     # name of corpus category. For example: 'sew'
N_DIS = int(args[4])    # number of dissimilar embeddings to select
# python3 cosine.py 'music_train' 'games_train' 'games' 10000
# python3 cosine.py 'music_train' 'sew_train' 'sew' 10000


# Load base corpus file and split into X and y
data_1 = csv_loader('../data/interim/' + FILE_1 + '.csv')
#data_1 = data_1[0:10]
X_1, y_1 = data_1[['review']], data_1[['sentiment']]

# Load Second corpus CSV file and split into X and y
data_2 = csv_loader('../data/interim/' + FILE_2 + '.csv')
#data_2 = data_2[0:10]
X_2, y_2 = data_2[['review']], data_2[['sentiment']]

# define stop words
stop_words = set(stopwords.words('english'))

# Tokenize each review and lowercase everything
corp_1 = []
for i in range(len(X_1)): 
    row = X_1.iloc[i]['review']
    token_review = word_tokenize(row)
    filtered = [w.lower() for w in token_review if not w.lower() in stop_words]
    corp_1.append(filtered)

# dictionary of tokens
dictionary = gensim.corpora.Dictionary(corp_1)

# make BOW
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in corp_1]

# TFIDF to downplay frequent words
tf_idf = gensim.models.TfidfModel(corpus)

# building the index
sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))

# tokenize second corpus, lowercase, and compute similarity
corp_2 = []
avg_sims = [] # array of averages
print("Starting Sentence Comparison")
for i in tqdm(range(1,len(X_2))): # start at 1 because 0 is "Review"
    row = X_2.iloc[i]['review']
    token_review = word_tokenize(row)
    filtered = [w.lower() for w in token_review if not w.lower() in stop_words]
    query_doc_bow = dictionary.doc2bow(filtered) # update an existing dictionary and create bag of words
    corp_2.append(filtered)
    
    # perform a similarity query against the corpus
    query_doc_tf_idf = tf_idf[query_doc_bow]
    
    doc_sim = sims[query_doc_tf_idf]
    #print(f"Comparing Similarity: {doc_sim}")
    
    # Average Similarity score
    sum_of_sims =(np.sum(doc_sim, dtype=np.float32)) # find average similarity
    sim_ave = sum_of_sims/len(corp_1)                # round (removed the rounding to see if that changes things)
    avg_sims.append((sim_ave, i))                    # append (similarity, sentence index)
        
print("Done!")
