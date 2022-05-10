import numpy as np
import json
import pandas as pd
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d%m%Y_%H%M%S")

def get_predictions(N, PROB_FILE):
    # Get predictions from the probabilities
    threshold = N
    probs = []
    with open('../data/probabilities/' + PROB_FILE, 'r') as f:
        temp = (f.read().split(','))
        for i in temp[:-1]:
            a = i.split(' ')
            #print(len(a))
            t = []
            for u in a:
                if len(u) > 1:
                    t.append(u)
            #print(len(t))
            probs.append(t)

    probs = np.array(probs)
    #print(probs)
    preds = np.where(probs[:, 1] > threshold, 1, 0)

    # Number of tweets predicted non-negative
    #print("Number of reviews predicted positive: ", preds.sum())

    # convert 0s and 1s into strings
    y_hat = []

    for i in preds:
        if i:
            y_hat.append('positive')
        else: y_hat.append('negative')

    # for export, open the test data and append predicted sentiment
    test_pred = pd.read_json( '../data/raw/music_reviews_test_masked.json.gz', lines=True)
    test_pred['sentiment'] = y_hat
    
    # Write in a way that codalabs will accept. Thanks to Nicola.
    new = test_pred.to_dict('records')
    test_json=[json.dumps(i)+'\n' for i in new]
    with open ('../data/predictions/pickle_music_reviews_test_'+'_ALL_ALL_ALL'+'.json', 'w') as file:
        file.writelines(test_json)


if __name__ == '__main__':
    import sys
    args = sys.argv
    #print(args)
    THRESHOLD = args[1]  # threshold (between 0.1 and 1)
    PROB = args[2]       # probabilities file
    #TEST_DATA = args[3] # path to test data
    #PRED = args[4]      # path to save prediction file
    get_predictions(THRESHOLD, PROB)