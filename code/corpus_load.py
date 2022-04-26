import gzip
import json
import csv
import re
import sys
import pandas as pd
import numpy as np

args = sys.argv
if len(args) < 2:
    print("You forgot to enter the file")
CORPUS = args[1]  # raw corpus file. For example: # music_reviews_train.json.gz
#CSV = args[2]     # name to save interim CSV file. For example: # games.csv
# = args[3]  # 

#######################################################
review_keys = set(['image', 'vote'])

def loader(PATH):
    review_list = []
    for line in gzip.open(PATH):
        review_data = json.loads(line)
        temp = {'image': 0, # set image binary to 0
                'reviewText' : '<NULL>',
                'summary' : '<NULL>',
                'style' : '<NULL>',
                'vote' : 0} # set votes to zero, will be overwritten if there are upvotes
        for key in review_data:
            if key == 'image':
                temp[key] = 1 # if there is an image present, set the binary to 1
            else:
                if key == 'sentiment':
                    if review_data[key] == 'positive':
                        temp[key] = 1
                    elif review_data[key] == 'negative': 
                        temp[key] = 0                    
                else:
                    review_keys.update([key])
                    temp[key] =  str(review_data[key])
        review_list.append(temp)
    return review_list 
    
def set_making(data, test = False):
    """Function to separate data into X and y.
    Input: 
    - A list of dictionaries of reviews. 
    Output:
    - X: a list of concatenated summary and text of reviews.
    - y: a list of corresponding sentiment labels.
    """
    X = []
    y = []
    for i in data:
        X.append(i['summary'] + ' ' +i['reviewText'])
        if test == False and 'sentiment' in i:
            y.append(i['sentiment'])
        elif test == False and 'overall' in i:
            rating = float(i['overall'])
            y.append(int(rating))
    if test:
        return X
    else: return X, y
    
def csv_loader(PATH):
    text = pd.read_csv(PATH) 
    return text    

def splitter(L):
    X = []
    y = []
    for i in L:
        X.append(i[0])
        y.append(int(i[1]))
    return X, y

#####################################################
# Load Raw Data
raw_file = '../data/raw/' + CORPUS
raw = loader(raw_file)

# Make sets of Review Text and Star Rating
X, y = set_making(raw)

# Convert to dataframe
data = pd.DataFrame(list(zip(X,y)), columns=['review','sentiment'])

#######################################################
# Remove 3's
# Get indexes where name column has value 3
indexNames = data[data['sentiment'] == 3].index

# Delete these row indexes from dataFrame
data.drop(indexNames , inplace=True)

#######################################################
# Dictionary to map star to "0" (negative) or "1" (positive)
label_dict = {1:0, 2:0, 4:1, 5:1}
data['lable'] = data['sentiment'].map(label_dict)
data.drop('sentiment', axis=1, inplace=True)

print(data.head())

#######################################################
# Train, Dev, Test, Split



#######################################################
# save to csv
#data.to_csv('../data/interim/' + CSV, index=False)

# Load Interim CSV file
#data = csv_loader('../data/interim/' + CSV)

