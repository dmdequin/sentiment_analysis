"""Script to Compute KL-Divergence
Between 2 Corpus
"""
import time
import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

start_time = time.time()

args = sys.argv
if len(args) < 2:
    print("You forgot something")
FILE_1 = args[1]  # name of interim csv file. For example: # games_train
FILE_2 = args[2]  # name of comparison interim csv file. For example: # sew_train
# python3 kl_divergence.py music_train games_train

def csv_loader(PATH):
    text = pd.read_csv(PATH, names=['review','sentiment']) 
    return text
    
def kl_divergence(a, b):
    sums = 0
    for i in range(len(a)):
        if b[i]!=0.01:
            sums += a[i] * np.log(a[i]/b[i])
    return sums
    
# Load Interim CSV file and split into X and y
data_1 = csv_loader('../data/interim/' + FILE_1 + '.csv')
X_1, y_1 = data_1[['review']], data_1[['sentiment']]
#X_1 = X_1[0:1000]

# Load Interim CSV file and split into X and y
data_2 = csv_loader('../data/interim/' + FILE_2 + '.csv')
X_2, y_2 = data_2[['review']], data_2[['sentiment']]
#X_2 = X_2[0:1000]

stop_words = list(stopwords.words('english'))+list(string.punctuation)

# Tokenize each review and lowercase everything
corp_1 = []
for i in range(len(X_1)): 
    row = X_1.iloc[i]['review']
    token_review = word_tokenize(row)
    filtered = [corp_1.append(w.lower()) for w in token_review if not w.lower() in stop_words]

print("Finished Tokenizeing Corp 1")

# List of all Words    
all_words = list(set(corp_1))

corp_2 = []
for i in range(1,len(X_2)): 
    row = X_2.iloc[i]['review']
    token_review = word_tokenize(row)
    filtered = [corp_2.append(w.lower()) for w in token_review if not w.lower() in stop_words]

# Add words from corp 2 to list
all_words = all_words + corp_2
all_words = list(set(all_words))

print("Finished Tokenizeing Corp 2")
############################################################

# Word Count Dictionaries for Corpses
# Dictionary of word to index, and index to word
word_to_idx = {}
idx_to_word = {}

# size of corpses
count_corp1 = len(corp_1)
count_corp2 = len(corp_2)

# Initialize Dictionaries for both corpus
dict_1 = {}
for i in range(len(all_words)):
    word_to_idx[all_words[i]] = i
    idx_to_word[i] = all_words[i]
    dict_1[word_to_idx[all_words[i]]] = 0.01 # add 0.01 as starter, to avoid 0 division
    
dict_2 = {}
for i in all_words:
    dict_2[word_to_idx[i]] = 0.01

# get word count for corpus 1
for i in corp_1:
    dict_1[word_to_idx[i]] += 1

# get word count for corpus 2
for i in corp_2:
    dict_2[word_to_idx[i]] += 1

print("Finished Word Counts")
#######################################################33

# Convert to Probabilities

# get word probabilites for corp 1
corp1_prob = {}
for k,v in dict_1.items():
    prob = v/count_corp1    
    corp1_prob[k] = prob
prob1_list = list(corp1_prob.values())

# get word probabilites for corp 2
corp2_prob = {}
for k,v in dict_2.items():
    prob2 = v/count_corp2
    corp2_prob[k] = prob2
prob2_list = list(corp2_prob.values())

print("Finished Probabilitities")

print("Computing KL-Divergence")

kld = kl_divergence(prob1_list, prob2_list)

print(f"KL-Divergence between corpses: {kld}")

print(f"Execution Time: {round(time.time() - start_time, 2)}")




