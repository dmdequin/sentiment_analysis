import gensim
from nltk.tokenize import word_tokenize, sent_tokenize

def tokenize(string):
    '''
    Input: A string, can contain multiple sentences.
    Output: A list of lists where each sublist is a sentence and the strings in it are the tokenised words.
    '''
    # list of strings
    sentences = sent_tokenize(string)
    # sentences = string.lower()

    # list of lists, words as tokens for corpus 1
    tokens = [[word.lower() for word in word_tokenize(text)] for text in sentences]
    # tokens = [word_tokenize(sentences)]

    return tokens

def make_dict(tokens):
    '''
    Input: A list of lists where each sublist is a sentence and the strings in it are the tokenised words.
    Output: returns indexed similarities
    '''
    dictionary = gensim.corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('workdir/', tf_idf[corpus], num_features=len(dictionary))
    return dictionary, tf_idf, sims


def compare(string_index, string_query):
    tokens = tokenize(string_index)
    dictionary, tf_idf, sims = make_dict(tokens)

    query = [w.lower() for w in word_tokenize(string_query)] # tokenise into strings
    # print(query)
    query_bow = dictionary.doc2bow(query) # bag of words for string being queried
    query_tf_idf = tf_idf[query_bow] # ??? 
    
    # get similarity scores between given string and indexes text
    sim_score =  sims[query_tf_idf]
    # print(len(sim_score), sim_score)

    # return the avarage sim between the queried string and all sentences in the indexed text - in the case review(s)
    return sum(sim_score) / len(sim_score)