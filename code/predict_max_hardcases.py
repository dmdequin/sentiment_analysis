import numpy as np
import json
import pandas as pd
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d%m_%H%M")

def get_predictions(PATH):
    # Get predictions from the probabilities
    probs = []
    with open('../data/probabilities/' + PATH, 'r') as f:
        temp = (f.read().split(','))
        for i in temp[:-1]:
            #print('i', i)
            a = i.split(' ')
            #print(len(a))
            t = []
            for u in a:
                if len(u) > 1:

                    t.append(float(u[1:-1]))
            #print((t))
            probs.append(t)
    #print(probs)
    probs = np.array(probs)
    #print(probs)
    preds = np.argmax(probs, axis=1)
    #print(preds)

    # Number of tweets predicted non-negative
    #print("Number of reviews predicted positive: ", preds.sum())

    # convert 0s and 1s into strings
    y_hat = []

    for i in preds:
        if i:
            y_hat.append('positive')
        else: y_hat.append('negative')
    # for export 
    test_pred = pd.read_json( '../data/raw/phase2_testData-masked.json.gz', lines=True)
    #test_pred = test_pred.iloc[0:100]
    test_pred['sentiment'] = y_hat
    #Write in a way that codalabs will accept. Thanks to Nicola.
    new = test_pred.to_dict('records')
    test_json=[json.dumps(i)+'\n' for i in new]
    with open ('../data/predictions/all_hardcases_predictions'+'.json', 'w') as file:
        file.writelines(test_json)


if __name__ == '__main__':
    import sys
    args = sys.argv
    #print(args)
    get_predictions(args[1])