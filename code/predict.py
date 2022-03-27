import numpy as np
import json
import pandas as pd
from datetime import datetime

now = datetime.now()
current_date_time = now.strftime("%d%m%Y_%H%M%S")

def get_predictions(N, PATH):
    # Get predictions from the probabilities
    threshold = N
    probs = []
    with open(PATH, 'r') as f:
        temp = (f.read().split(','))
        for i in temp[:-1]:
            a = i.split(' ')
            probs.append([a[0][1:],a[1][:-1] ])

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

    # for export 
    test_pred = pd.read_json( '../data/raw/music_reviews_test_masked.json.gz', lines=True)
    #test_pred = test_pred.iloc[0:100]
    test_pred['sentiment'] = y_hat
    #Write in a way that codalabs will accept. Thanks to Nicola.
    new = test_pred.to_dict('records')
    test_json=[json.dumps(i)+'\n' for i in new]
    with open ('../data/predictions/pickle_music_reviews_test_'+current_date_time+'.json', 'w') as file:
        file.writelines(test_json)


if __name__ == '__main__':
    import sys
    args = sys.argv
    #print(args)
    get_predictions(args[1], args[2])